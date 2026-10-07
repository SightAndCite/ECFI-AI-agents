"""Build the paper figures from saved results.

IEEE conventions: vector PDF, 3.4 in single column / 7.0 in double column,
8 pt labels, recessive grid.  Every series carries a marker and a line style as
well as a hue, so the figures survive grayscale printing and colour-vision
deficiency -- the palette's tightest adjacent pair sits in the CVD floor band,
which is legal only with that secondary encoding.

No dual-axis plots: quantities on different scales get their own panel.

Usage:  python experiments/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecfi.linear import ILLUSTRATIVE, kappa_crit, spectral_radius, u_star  # noqa: E402
from ecfi.runner import FIGS, RESULTS, load_json  # noqa: E402

# Okabe-Ito subset, ordered so every adjacent pair clears the CVD floor.
PALETTE = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
MARKERS = ["o", "s", "^", "D"]
STYLES = ["-", "--", "-.", ":"]
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"

COL1, COL2 = 3.4, 7.0

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
    "axes.axisbelow": True, "axes.linewidth": 0.6,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "legend.frameon": False, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figs/{name}.pdf")


def exists(name):
    p = RESULTS / name
    return p if p.exists() else None


# --------------------------------------------------------------------------- #


def fig_theory():
    """The verification-congestion paradox in the illustrative regime."""
    p = ILLUSTRATIVE
    kc = kappa_crit(p)
    ks = np.linspace(0, 2.0, 201)
    rho = np.array([spectral_radius(p, k) for k in ks])
    us = np.array([u_star(p, k) for k in ks])
    us = us / us[0]

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.0))
    ax = axes[0]
    ax.plot(ks, us, color=PALETTE[1], ls=STYLES[1])
    ax.set_xlabel(r"verification gain $\kappa$")
    ax.set_ylabel(r"$u^*(\kappa)/u^*(0)$")
    ax.set_title("(a) static equilibrium improves", loc="left")
    ax.axvline(kc, color=MUTED, lw=0.6, ls=":")

    ax = axes[1]
    ax.plot(ks, rho, color=PALETTE[2], ls=STYLES[0])
    ax.axhline(1.0, color=MUTED, lw=0.6, ls="--")
    ax.axvline(kc, color=MUTED, lw=0.6, ls=":")
    ax.annotate(rf"$\kappa_{{\rm crit}}={kc:.3f}$", xy=(kc, 1.0),
                xytext=(kc - 0.85, 1.15), fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
    ax.set_xlabel(r"verification gain $\kappa$")
    ax.set_ylabel(r"$\rho(A(\kappa))$")
    ax.set_title("(b) dynamic stability is lost", loc="left")
    fig.tight_layout()
    save(fig, "fig_theory_paradox")


def fig_s1():
    path = exists("s1_gonogo.json")
    if not path:
        return
    d = load_json(path)
    labels = {"wait": "Wait", "resource_coupled": "Resource-coupled",
              "resource_coupled_nostore": "Resource-coupled, store off"}

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.1))
    for i, (arm, lab) in enumerate(labels.items()):
        if arm not in d["arms"]:
            continue
        per = d["arms"][arm]["per_level"]
        lv = sorted(float(k) for k in per)
        err = [per[str(k) if str(k) in per else f"{k}"]["u"][0] for k in lv]
        se = [per[str(k) if str(k) in per else f"{k}"]["u"][1] for k in lv]
        wt = [per[str(k) if str(k) in per else f"{k}"]["wait_p95"][0] for k in lv]
        axes[0].errorbar(lv, err, yerr=se, color=PALETTE[i], ls=STYLES[i],
                         marker=MARKERS[i], label=lab, capsize=1.5, elinewidth=0.6)
        axes[1].plot(lv, wt, color=PALETTE[i], ls=STYLES[i], marker=MARKERS[i],
                     label=lab)
    axes[0].set_xlabel("exogenous background load")
    axes[0].set_ylabel("objective error rate")
    axes[0].set_title("(a) correctness", loc="left")
    axes[0].legend(loc="upper left")
    axes[1].set_xlabel("exogenous background load")
    axes[1].set_ylabel("p95 verification wait")
    axes[1].set_title("(b) latency", loc="left")
    fig.tight_layout()
    save(fig, "fig_s1_gonogo")


def _sweep(regime):
    return exists(f"s3_sweep_{regime}.json")


def fig_s3_precursors():
    regimes = [r for r in ("steep", "shallow") if _sweep(r)]
    if not regimes:
        return
    fig, axes = plt.subplots(len(regimes), 3, figsize=(COL2, 2.1 * len(regimes)),
                             squeeze=False)
    for row, regime in enumerate(regimes):
        d = load_json(_sweep(regime))
        k_hat = float(d["kappa_hat_crit"])
        for i, (load, blob) in enumerate(sorted(d["by_load"].items())):
            pk = blob["per_kappa"]
            ks = sorted(float(k) for k in pk)
            x = np.array(ks) / k_hat
            get = lambda key: [pk[str(k)][key][0] for k in ks]  # noqa: E731
            axes[row][0].plot(x, get("margin"), color=PALETTE[i], ls=STYLES[i],
                              marker=MARKERS[i], label=rf"$\lambda$ factor {load}")
            axes[row][1].plot(x, get("acf1_u"), color=PALETTE[i], ls=STYLES[i],
                              marker=MARKERS[i])
            avail = pk[str(ks[0])]
            if "var_ratio_u" in avail:
                y = np.array(get("var_ratio_u"))
            elif "sd_u" in avail:
                sd = np.array(get("sd_u"))
                y = (sd / sd[0]) ** 2
            else:
                y = None
            if y is not None:
                axes[row][2].plot(x, y, color=PALETTE[i], ls=STYLES[i],
                                  marker=MARKERS[i])

        for j, (ylab, ttl) in enumerate([
            (r"$M = 1-\rho(\hat A)$", "stability margin"),
            (r"lag-1 autocorr. of $u_k$", "alternation"),
            (r"var$(u_k)$ / var$(u_k)|_{\kappa=0}$", "variance amplification"),
        ]):
            ax = axes[row][j]
            ax.axvline(1.0, color=MUTED, lw=0.7, ls="--")
            ax.set_xlabel(r"$\kappa / \hat\kappa_{\rm crit}$")
            ax.set_ylabel(ylab)
            ax.set_title(f"({'abcdef'[row * 3 + j]}) {regime}: {ttl}", loc="left")
        axes[row][0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_s3_precursors")


def fig_s3_ratio():
    path = exists("s3_endpoints.json")
    if not path:
        return
    d = load_json(path)
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    for i, (regime, blob) in enumerate(d.items()):
        loads, ratios = [], []
        for load, v in sorted(blob["by_load"].items()):
            if v.get("ratio_knee"):
                loads.append(float(load)); ratios.append(v["ratio_knee"])
        if loads:
            ax.plot(loads, ratios, color=PALETTE[i], ls=STYLES[i],
                    marker=MARKERS[i], label=regime)
            ax.annotate(regime, xy=(loads[-1], ratios[-1]),
                        xytext=(3, 0), textcoords="offset points",
                        fontsize=7, color=PALETTE[i], va="center")
    ax.axhline(1.0, color=MUTED, lw=0.7, ls="--")
    ax.text(0.455, 1.02, "perfect prediction", fontsize=6.5, color=MUTED)
    ax.set_xlabel("offered load factor")
    ax.set_ylabel(r"observed $\kappa^\ast$ / predicted $\hat\kappa_{\rm crit}$")
    ax.set_ylim(0, 1.25)
    fig.tight_layout()
    save(fig, "fig_s3_ratio")


def fig_s5():
    paths = [p for p in (exists("s5_mitigation_steep.json"),
                         exists("s5_mitigation_shallow.json")) if p]
    if not paths:
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.1))
    for i, path in enumerate(paths):
        d = load_json(path)
        gammas = sorted(float(g) for g in d["by_gamma"])
        err = [d["by_gamma"][str(g)]["error"][0] for g in gammas]
        p95 = [d["by_gamma"][str(g)]["wait_p95"][0] for g in gammas]
        marg = [d["by_gamma"][str(g)]["margin"][0] for g in gammas]
        axes[0].plot(p95, err, color=PALETTE[i], ls=STYLES[i], marker=MARKERS[i],
                     label=d["regime"])
        for g, x, y in zip(gammas, p95, err):
            if g in (gammas[0], gammas[len(gammas) // 2], gammas[-1]):
                axes[0].annotate(rf"$\gamma={g:g}$", xy=(x, y), xytext=(3, 3),
                                 textcoords="offset points", fontsize=6.5, color=MUTED)
        axes[1].plot(gammas, marg, color=PALETTE[i], ls=STYLES[i], marker=MARKERS[i],
                     label=d["regime"])
    axes[0].set_xlabel("p95 verification wait")
    axes[0].set_ylabel("objective error rate")
    axes[0].set_title("(a) quality-latency frontier", loc="left")
    axes[0].legend(loc="best")
    axes[1].set_xscale("symlog")
    axes[1].set_xlabel(r"congestion-aware damping $\gamma$")
    axes[1].set_ylabel(r"$M = 1-\rho(\hat A)$")
    axes[1].set_title("(b) stability gain", loc="left")
    fig.tight_layout()
    save(fig, "fig_s5_frontier")


def fig_s4():
    path = exists("s4_strategic_steep.json")
    if not path:
        return
    d = load_json(path)
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    fields = [("q", "queue occupancy"), ("wait_p95", "p95 wait"), ("tokens", "effort served")]
    eps = f"{max(float(e) for e in d['epsilons']):.2f}"
    for i, (f, lab) in enumerate(fields):
        xs, ys = [], []
        for gf, blob in sorted(d["by_gain"].items()):
            by_eps = blob["by_epsilon"]
            key = next((k for k in by_eps if abs(float(k) - float(eps)) < 1e-9), None)
            if key is None:
                continue
            xs.append(blob["kappa"] / d["kappa_hat_crit"])
            ys.append(by_eps[key]["a_e"][f][0])
        if xs:
            ax.plot(xs, ys, color=PALETTE[i], ls=STYLES[i], marker=MARKERS[i], label=lab)
    ax.axhline(1.0, color=MUTED, lw=0.7, ls="--")
    ax.set_xlabel(r"$\kappa / \hat\kappa_{\rm crit}$")
    ax.set_ylabel(r"$\mathcal{A}_E$ (normalised)")
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, "fig_s4_amplification")


def main():
    fig_theory()
    fig_s1()
    fig_s3_precursors()
    fig_s3_ratio()
    fig_s4()
    fig_s5()


if __name__ == "__main__":
    main()
