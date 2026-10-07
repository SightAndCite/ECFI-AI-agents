"""The S6 verifier pool: deadline and queue accounting, no model needed."""

import importlib.util
import threading
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "s6", Path(__file__).resolve().parent.parent / "experiments" / "s6_real_agents.py")
s6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s6)


def test_pool_deadline_and_queue():
    pool = s6.VerifierPool(1)
    release = threading.Event()
    pool.submit(release.wait)                       # occupy the only slot
    out, wait, timed_out = pool.request(lambda: 1, deadline=0.05)
    assert timed_out and out is None and wait >= 0.05
    assert pool.waiting == 0                        # abandoned request left the queue

    res = {}
    t = threading.Thread(target=lambda: res.update(r=pool.request(lambda: 42)))
    t.start()
    while pool.waiting != 1:                        # WAIT mode: queued, not abandoned
        pass
    assert pool.q_obs() == 1.0
    release.set()
    t.join(5)
    assert res["r"][0] == 42 and res["r"][2] is False
    pool.drain()


def test_closed_loop_windows(monkeypatch):
    """S7 loop bookkeeping with the model call stubbed out."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    import s7_real_loop as s7

    monkeypatch.setattr(s7.s6, "verify", lambda model, a, b: (a * b, 0.0))  # perfect verifier
    monkeypatch.setattr(s7, "WARMUP", 2)
    bank = [{"a": 2, "b": 3, "hard": False, "primary": 0}]                  # always wrong unverified
    ws = s7.run_loop(bank, 1, 0.0, 6, "stub", vbar=lambda k: 1.0, rate=2000.0)
    assert len(ws) == 4 and ws.u.max() == 0.0 and ws.v.min() == 1.0        # all verified, all fixed
    ws = s7.run_loop(bank, 1, 0.0, 6, "stub", vbar=lambda k: 0.0, rate=2000.0)
    assert ws.u.min() == 1.0 and ws.tokens.sum() == 0.0                    # none verified, all wrong
