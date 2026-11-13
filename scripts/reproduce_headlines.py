"""Print the headline numbers quoted in the README from the cached training runs, and write two figures to figures/.

Usage (from the repository root, after `pip install -e .`):
    python scripts/reproduce_headlines.py

No training happens here (that takes hours). The cached runs in results/ were produced by the code in
notebooks/ and src/mag_diffusion_pinn/. To retrain a single run, see the README.
"""
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mag_diffusion_pinn.inverse import crb_theta, make_probes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
THETA_TRUE = 0.6


def load(*names):
    out = {}
    for n in names:
        with open(os.path.join(RES, n + ".pkl"), "rb") as f:
            out.update(pickle.load(f))
    return out


def stats(v):
    v = np.asarray(v, float)
    return f"median {np.median(v):.2e}  [min {v.min():.1e}, max {v.max():.1e}]  (n={len(v)})"


def main():
    os.makedirs(FIG, exist_ok=True)
    m2 = load("m2_runs")
    runs = load("m3_A", "m3_R", *[f"m3b_{i}" for i in range(1, 8)])
    rel = lambda prob, cfg, seeds: [runs[(prob, cfg, s)]["metrics"]["rel_l2"] for s in seeds]
    six = range(6)

    plain_A = [m2[("A", "float32", s)]["metrics"]["rel_l2"] for s in (0, 1, 2)] + rel("A", "plain", (3, 4, 5))
    series = {"A": {"plain": plain_A, "hard": rel("A", "hard", six), "manual weights": rel("A", "w_manual", six),
                    "capped adaptive": rel("A", "w_adapt_cap", six)},
              "R": {"plain": rel("R", "plain", six), "hard": rel("R", "hard", six)}}

    print("FORWARD PROBLEMS: relative L2 error over the (xi, tau) grid, six seeds, float32")
    for prob, d in series.items():
        print(f"  problem {prob}:")
        for name, v in d.items():
            print(f"    {name:18s} {stats(v)}")
    print(f"  R: hard vs plain median ratio = {np.median(series['R']['plain']) / np.median(series['R']['hard']):.0f}x")
    print(f"  A: plain vs hard median ratio = {np.median(plain_A) / np.median(series['A']['hard']):.1f}x")

    inv = load("m4_0", "m4_1")
    print("\nINVERSE PROBLEM: true theta = 0.6, PINN starts at 1.0")
    print(f"  noise-free (3 probes): theta = {inv[('p3', 0.0, 0)]['theta']:.5f}")
    xi, t = make_probes([0.2, 0.5, 0.8])
    for s in (0.005, 0.01, 0.02, 0.05):
        th = np.array([inv[("p3", s, r)]["theta"] for r in range(3)])
        print(f"  3 probes, noise {100 * s:.1f}%: theta = {', '.join(f'{x:.4f}' for x in th)}; "
              f"errors {', '.join(f'{100 * (x - THETA_TRUE) / THETA_TRUE:+.2f}%' for x in th)}; "
              f"Cramer-Rao std {crb_theta(xi, t, THETA_TRUE, s):.4f} ({100 * crb_theta(xi, t, THETA_TRUE, s) / THETA_TRUE:.2f}% of theta)")

    # figure 1: forward errors
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (prob, d) in zip(axes, series.items()):
        for x, (name, v) in enumerate(d.items()):
            ax.semilogy([x] * len(v), v, "o", color="C0", alpha=0.7)
            ax.semilogy([x - 0.25, x + 0.25], [np.median(v)] * 2, "k-", lw=1.5)
        ax.set_xticks(range(len(d))); ax.set_xticklabels(list(d), fontsize=8)
        ax.set_xlim(-0.6, len(d) - 0.4); ax.grid(alpha=0.3)
        ax.set_title(f"Problem {prob}: relative L2 error, six seeds (bar: median)", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "forward_errors.png"), dpi=140); plt.close()

    # figure 2: inverse noise sensitivity
    sig = [0.005, 0.01, 0.02, 0.05]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.loglog(sig, [crb_theta(xi, t, THETA_TRUE, s) for s in sig], "k-", label="Cramer-Rao std (best possible)")
    for s in sig:
        for r in range(3):
            ax.loglog([s], [abs(inv[("p3", s, r)]["theta"] - THETA_TRUE)], "o", color="C0", alpha=0.75)
    ax.loglog([], [], "o", color="C0", label="PINN |error|, one point per run")
    ax.set_xlabel("noise level (fraction of B0)"); ax.set_ylabel("error in theta"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8); ax.set_title("Recovering theta from 3 probes", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "inverse_noise.png"), dpi=140); plt.close()
    print(f"\nwrote {os.path.join(FIG, 'forward_errors.png')} and {os.path.join(FIG, 'inverse_noise.png')}")


if __name__ == "__main__":
    sys.exit(main())
