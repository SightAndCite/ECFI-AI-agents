"""Discrete-event simulation testbed for ECFI (Section VI).

A population of ``N`` agents processes tasks that arrive as a Poisson stream
and carry objective ground truth.  Verification requests contend for a shared
pool of ``C`` verifier servers with lognormal service times and a FIFO queue.
Two couplings close the loop:

*  **Epistemic coupling (the b-mechanism).**  Tasks whose final answer is wrong
   are written to a shared context store.  Each new task samples that store
   with probability ``p_dep``; consuming an erroneous artifact raises its error
   probability by ``delta_dep``.  Disabling the store sets ``b ~ 0`` *by
   construction* and is the internal negative control.

*  **Resource coupling.**  Each task carries a verification deadline ``D``.  In
   ``RESOURCE_COUPLED`` mode an expired deadline triggers a pre-specified
   fallback -- skipped verification plus a weaker model with error multiplier
   ``m > 1``.  In ``WAIT`` mode deadlines are disabled and all planned
   verification completes, so pure queueing raises latency without changing
   correctness.

Verification efficacy is a saturating function of effort,

    q_fix(v) = 1 / (1 + exp(-beta (v - v_half))),

i.e. a cheap check rarely catches an error and a thorough one usually does.
This is what makes over-correction possible and is the nonlinearity the local
linearization of Section II approximates.

Aggregating over windows of ``W`` completed tasks yields the discrete-time
observables ``(u_k, q_k, v_k)`` the local model is written in.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import simpy

from .policies import make_policy
from .schedules import Schedule, constant

__all__ = ["SimConfig", "WindowSeries", "run_sim", "WAIT", "RESOURCE_COUPLED"]

WAIT = "wait"
RESOURCE_COUPLED = "resource_coupled"


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


@dataclass
class SimConfig:
    """Default simulation parameters (Table II of the paper)."""

    # --- environment ---
    n_agents: int = 50                 # N
    n_verifiers: int = 8               # C
    mu: float = 1.0                    # verifier service rate per unit effort
    service_cv: float = 0.6            # lognormal coefficient of variation
    load_factor: float = 0.55          # lambda * v_bar / (C mu) at nominal effort

    # --- epistemics ---
    p0: float = 0.12                   # nominal error probability
    agent_spread: float = 0.03         # per-agent competence sd
    sigma_conf: float = 0.05           # confidence miscalibration sd

    # --- b-mechanism (shared context store) ---
    store_enabled: bool = True         # False => b ~ 0 by construction
    store_size: int = 200
    p_dep: float = 0.80                # task dependency probability
    n_dep: int = 4                     # artifacts a dependent task consumes
    delta_dep: float = 0.45            # error-prob boost at fully bad context

    # --- verification efficacy ---
    beta: float = 2.0                  # steepness of q_fix
    v_half: float = 1.30               # effort at which q_fix = 0.5

    # --- policy ---
    kappa: float = 0.0
    kappa_local: float = 0.0           # per-task gain on (u_hat - system signal)
    gamma: float = 0.0                 # congestion-aware damping (Section V-5)
    v_bar: float = 1.30                # nominal effort (sits at v_half)
    v_max: float = 4.0
    u_ref: float = 0.12                # initial nominal uncertainty (set-point)
    u_ref_adaptive: bool = True        # track the operating point with an EWMA
    ref_tau: float = 50.0              # set-point time constant, in windows
    signal_noise: float = 0.010        # monitoring noise on the u_k observable

    # --- resource coupling ---
    mode: str = RESOURCE_COUPLED
    deadline: float = 2.0              # D, on time waiting for a verifier
    m_fallback: float = 2.0            # fallback error multiplier

    # --- windowing ---
    W: int = 200                       # window size in completed tasks
    n_windows: int = 260
    warmup_windows: int = 60

    # --- exogenous excitation (all optional) ---
    bg_load: Optional[Schedule] = None       # background job intensity
    ambiguity: Optional[Schedule] = None     # P(task is ambiguous)
    kappa_sched: Optional[Schedule] = None   # overrides kappa when present
    v_bar_sched: Optional[Schedule] = None   # randomized *nominal* effort.
    # Dithering v_bar is the intervention that identifies c and f: it moves
    # verification effort independently of uncertainty, which dithering kappa
    # cannot do -- near the operating point u - u_ref is small, so a change in
    # kappa barely moves v and leaves the design near-collinear.
    amb_u_boost: float = 0.22          # uncertainty added by an ambiguous task
    amb_true_boost: float = 0.10       # true error added by an ambiguous task
    bg_service: float = 1.0            # mean service demand of a background job

    seed: int = 0

    def lam(self) -> float:
        """Task arrival intensity implied by the target utilisation."""
        return self.load_factor * self.n_verifiers * self.mu / max(self.v_bar, 1e-9)

    def to_dict(self) -> Dict:
        d = asdict(self)
        for k in ("bg_load", "ambiguity", "kappa_sched"):
            d[k] = None if getattr(self, k) is None else "schedule"
        return d


@dataclass
class WindowSeries:
    """Windowed observables and the exogenous inputs that produced them."""

    u: np.ndarray            # objective error rate
    q: np.ndarray            # mean verifier-queue occupancy at request, / C
    v: np.ndarray            # mean requested verification effort
    kappa: np.ndarray        # gain in force during the window
    bg: np.ndarray           # background-load intensity during the window
    amb: np.ndarray          # ambiguous-task fraction during the window
    wait: np.ndarray         # mean verification wait
    wait_p95: np.ndarray     # tail latency
    fallback: np.ndarray     # fraction of tasks that hit the deadline
    tokens: np.ndarray       # total verification effort served (cost proxy)
    t_end: np.ndarray        # simulation time at window close
    meta: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.u)

    def deviations(self):
        """(u, q, v) as deviations from their own means -- what the local model uses."""
        return (
            self.u - self.u.mean(),
            self.q - self.q.mean(),
            self.v - self.v.mean(),
        )

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "u": self.u, "q": self.q, "v": self.v, "kappa": self.kappa,
            "bg": self.bg, "amb": self.amb, "wait": self.wait,
            "wait_p95": self.wait_p95, "fallback": self.fallback,
            "tokens": self.tokens, "t_end": self.t_end,
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _lognormal_params(mean: float, cv: float):
    """Return (mu, sigma) of the underlying normal for a target mean and CV."""
    sigma2 = math.log(1.0 + cv * cv)
    return math.log(max(mean, 1e-12)) - 0.5 * sigma2, math.sqrt(sigma2)


def _q_fix(v: float, beta: float, v_half: float) -> float:
    """Probability that verification at effort v catches an existing error."""
    return 1.0 / (1.0 + math.exp(-beta * (v - v_half)))


# --------------------------------------------------------------------------- #
# the simulation
# --------------------------------------------------------------------------- #


class _Sim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.env = simpy.Environment()
        self.pool = simpy.Resource(self.env, capacity=int(cfg.n_verifiers))
        # Fail here, with a readable message, rather than several thousand
        # events later inside simpy's Request constructor: a mismatched or
        # half-updated install shows up as a confusing AttributeError deep in
        # the library, and this turns it into a statement about the install.
        if not isinstance(self.pool, simpy.Resource):  # pragma: no cover
            raise TypeError(
                f"verifier pool is {type(self.pool).__name__}, not simpy.Resource "
                f"-- the installed ecfi and simpy are inconsistent. "
                f"Run: python experiments/diagnose.py"
            )

        # shared context store: a bounded record of recent outcomes
        self.store: deque = deque(maxlen=cfg.store_size)

        # per-agent competence offsets
        self.agent_off = self.rng.normal(0.0, cfg.agent_spread, size=cfg.n_agents)

        self.bg_load = cfg.bg_load or constant(0.0)
        self.ambiguity = cfg.ambiguity or constant(0.0)

        # system-level epistemic signal, updated at each window close, and the
        # slow set-point the policy measures deviations against.  The local
        # model of Section II is a linearization *around an operating point*;
        # the EWMA set-point is that operating point, tracked with a time
        # constant far longer than the window so it cannot mask window-scale
        # dynamics while it keeps nominal effort at v_bar.
        self.u_signal: float = cfg.u_ref
        self.u_ref_dyn: float = cfg.u_ref

        # accumulators for the window currently being filled
        self._reset_window()
        self.windows: List[dict] = []
        self.n_target = cfg.n_windows
        self.done = self.env.event()

        _, self._svc_sigma = _lognormal_params(1.0, cfg.service_cv)

    # -- window bookkeeping -------------------------------------------------

    def _reset_window(self):
        self._w_err: List[int] = []
        self._w_uhat: List[float] = []
        self._w_q: List[float] = []
        self._w_v: List[float] = []
        self._w_wait: List[float] = []
        self._w_fb: List[int] = []
        self._w_amb: List[int] = []
        self._w_kappa: List[float] = []
        self._w_bg: List[float] = []
        self._w_tokens: float = 0.0

    def _close_window(self):
        cfg = self.cfg
        wait = np.asarray(self._w_wait, dtype=float)
        self.windows.append(
            {
                "u": float(np.mean(self._w_err)),
                "q": float(np.mean(self._w_q)),
                "v": float(np.mean(self._w_v)),
                "kappa": float(np.mean(self._w_kappa)),
                "bg": float(np.mean(self._w_bg)),
                "amb": float(np.mean(self._w_amb)),
                "wait": float(wait.mean()) if wait.size else 0.0,
                "wait_p95": float(np.percentile(wait, 95)) if wait.size else 0.0,
                "fallback": float(np.mean(self._w_fb)),
                "tokens": float(self._w_tokens),
                "t_end": float(self.env.now),
            }
        )
        # The system-level epistemic signal the policy acts on next window.
        # Section II writes u_k as system-level *error*, so the monitoring
        # observable is the measured correctness of completed work, seen
        # through observation noise -- not the agents' pre-verification
        # confidence, which lags the outcome by a further window and turns the
        # loop into a two-step map.
        self.u_signal = float(
            np.mean(self._w_err)
            + (self.rng.normal(0.0, self.cfg.signal_noise)
               if self.cfg.signal_noise > 0 else 0.0)
        )
        if self.cfg.u_ref_adaptive:
            w = 1.0 / max(self.cfg.ref_tau, 1.0)
            self.u_ref_dyn = (1.0 - w) * self.u_ref_dyn + w * self.u_signal
        self._reset_window()
        if len(self.windows) >= self.n_target and not self.done.triggered:
            self.done.succeed()

    def _record(self, wrong, u_hat, q_obs, v, wait, fb, amb, kappa, bg, served_effort):
        self._w_err.append(int(wrong))
        self._w_uhat.append(u_hat)
        self._w_q.append(q_obs)
        self._w_v.append(v)
        self._w_wait.append(wait)
        self._w_fb.append(int(fb))
        self._w_amb.append(int(amb))
        self._w_kappa.append(kappa)
        self._w_bg.append(bg)
        self._w_tokens += served_effort
        if len(self._w_err) >= self.cfg.W:
            self._close_window()

    # -- processes ----------------------------------------------------------

    def service_time(self, effort: float) -> float:
        """Lognormal service time with mean ``effort / mu``."""
        mean = max(effort, 1e-9) / self.cfg.mu
        mu_n, sig_n = _lognormal_params(mean, self.cfg.service_cv)
        return float(self.rng.lognormal(mu_n, sig_n))

    def queue_obs(self) -> float:
        """Congestion observable: waiting requests per verifier server."""
        return len(self.pool.queue) / self.cfg.n_verifiers

    def arrivals(self):
        cfg = self.cfg
        lam = cfg.lam()
        while True:
            yield self.env.timeout(self.rng.exponential(1.0 / lam))
            self.env.process(self.task())

    def background(self):
        """Exogenous load: jobs that occupy verifiers but carry no epistemics."""
        while True:
            rate = self.bg_load.at(self.env.now)
            if rate <= 0.0:
                yield self.env.timeout(0.25)
                continue
            yield self.env.timeout(self.rng.exponential(1.0 / rate))
            self.env.process(self.bg_job())

    def bg_job(self):
        with self.pool.request() as req:
            yield req
            yield self.env.timeout(self.service_time(self.cfg.bg_service))

    def task(self):
        cfg = self.cfg
        env = self.env
        rng = self.rng
        t0 = env.now

        kappa = cfg.kappa if cfg.kappa_sched is None else cfg.kappa_sched.at(t0)
        v_bar = cfg.v_bar if cfg.v_bar_sched is None else cfg.v_bar_sched.at(t0)
        policy = make_policy(kappa, cfg.gamma, v_bar, self.u_ref_dyn, cfg.v_max)

        agent = int(rng.integers(cfg.n_agents))
        p_err = cfg.p0 + self.agent_off[agent]

        # --- epistemic coupling: consume shared artifacts ------------------
        # A dependent task reads n_dep artifacts from the shared context store
        # and its error probability rises with the erroneous fraction among
        # them.  Reading several artifacts rather than one keeps the coupling
        # mean unchanged while cutting the sampling variance that would
        # otherwise dominate a window of W tasks.
        if cfg.store_enabled and self.store and rng.random() < cfg.p_dep:
            n = len(self.store)
            idx = rng.integers(n, size=cfg.n_dep)
            bad = sum(1 for i in idx if self.store[int(i)])
            p_err += cfg.delta_dep * (bad / cfg.n_dep)

        # --- ambiguity injection (identification bursts / strategic tasks) --
        amb_p = self.ambiguity.at(t0)
        is_amb = amb_p > 0.0 and rng.random() < amb_p
        u_extra = 0.0
        if is_amb:
            p_err += cfg.amb_true_boost
            u_extra = cfg.amb_u_boost

        p_err = min(max(p_err, 0.0), 0.98)
        wrong = rng.random() < p_err

        # imperfectly calibrated self-assessment
        u_hat = min(max(p_err + u_extra + rng.normal(0.0, cfg.sigma_conf), 0.0), 1.0)

        # The gain acts on the *system-level* windowed signal u_k, exactly as
        # v_k = kappa u_k in Section II: verification policy responds to the
        # measured epistemic state of the previous window, which is what gives
        # the closed loop its one-window delay.  kappa_local adds an optional
        # per-task component on top.
        q_obs = self.queue_obs()
        v = policy.effort(self.u_signal, q_obs)
        if cfg.kappa_local:
            v = min(max(v + cfg.kappa_local * (u_hat - self.u_signal), 0.0), cfg.v_max)

        wait = 0.0
        fallback = False
        served = 0.0

        if v > 1e-9:
            # The context-manager form is the one simpy guarantees: on exit an
            # untriggered request is removed from the queue and a granted one is
            # released -- whichever branch we leave through, and on an exception
            # too.  Doing that bookkeeping by hand leaks a verifier slot on any
            # path a test does not happen to exercise, and a leaked slot shows up
            # much later as an unexplained queue that never drains.
            with self.pool.request() as req:
                if cfg.mode == RESOURCE_COUPLED:
                    result = yield req | env.timeout(cfg.deadline)
                    granted = req in result
                else:  # WAIT: no deadline, all planned verification completes
                    yield req
                    granted = True

                if granted:
                    wait = env.now - t0
                    yield env.timeout(self.service_time(v))
                    served = v
                else:
                    fallback = True

            if fallback:
                # skipped verification + weaker model
                wrong = rng.random() < min(p_err * cfg.m_fallback, 0.98)
            elif wrong and rng.random() < _q_fix(v, cfg.beta, cfg.v_half):
                wrong = False

        if cfg.store_enabled:
            self.store.append(bool(wrong))

        self._record(
            wrong, u_hat, q_obs, v, wait, fallback, is_amb, kappa,
            self.bg_load.at(t0), served,
        )

    def run(self) -> WindowSeries:
        self.env.process(self.arrivals())
        self.env.process(self.background())
        self.env.run(until=self.done)

        cfg = self.cfg
        w = self.windows[cfg.warmup_windows:]
        if not w:
            raise RuntimeError("no windows left after warm-up; raise n_windows")
        get = lambda k: np.array([x[k] for x in w], dtype=float)  # noqa: E731
        return WindowSeries(
            u=get("u"), q=get("q"), v=get("v"), kappa=get("kappa"),
            bg=get("bg"), amb=get("amb"), wait=get("wait"),
            wait_p95=get("wait_p95"), fallback=get("fallback"),
            tokens=get("tokens"), t_end=get("t_end"),
            meta={"config": cfg.to_dict(), "n_windows_raw": len(self.windows)},
        )


def run_sim(cfg: SimConfig) -> WindowSeries:
    """Run one replication and return its windowed observables."""
    return _Sim(cfg).run()
