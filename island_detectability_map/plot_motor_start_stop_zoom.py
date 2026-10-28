"""Zooms on the residuals at the start and stop of the motor run in plot_motor_start_imbalance.py
(EXPLORATORY figures). Question: do B/C residuals show something at the stop (~22 s) and
A/B at the start? Residuals are shown minus their median in the 2 s before the window's event,
on an EXPANDED scale (stated on the axis). Run from the repo root.

Origin: verification@b066f888
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from island_detectability_map import phases
from island_detectability_map.plot_motor_start_imbalance import COL, NAME, RES, pick_onset


def main(data_dir: Path, out_dir: Path):
    z = np.load(data_dir / "grizzly_head6.npz"); fs = float(z["fs"])
    n = min(len(z[f"C{k}MAG"]) for k in (1, 2, 3))
    I = np.vstack([z[f"C{k}MAG"][:n].astype(np.float64) for k in (1, 2, 3)])
    on = pick_onset(I, fs)
    S = I.sum(axis=0); d = np.convolve(S, np.ones(12) / 12, "same")
    a = on + int(15 * fs); stop = a + int(np.argmin(np.diff(d[a:on + int(30 * fs)])))
    R = np.vstack([phases.imbalance(I, k) for k in range(3)])
    out = []
    for tag, ev, (lo, hi), lab in (("start", on, (-1.0, 1.5), "start-up onset"),
                                   ("stop", stop, (-2.0, 2.0), "stop (steepest fall of the summed current)")):
        i0, i1 = ev + int(lo * fs), ev + int(hi * fs); tt = (np.arange(i0, i1) - on) / fs
        pre = slice(ev - int(2 * fs), ev - int(0.1 * fs))
        fig, ax = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
        for k in range(3):
            ax[0].plot(tt, I[k, i0:i1], color=COL[k], lw=0.8, marker="o", ms=2.5, label=f"phase {NAME[k]}")
            ax[1].plot(tt, R[k, i0:i1] - np.median(R[k, pre]), color=COL[k], lw=0.8, marker="o", ms=2.5, label=RES[k])
        for a_ in ax:
            a_.axvline((ev - on) / fs, color="k", lw=0.6, ls=":"); a_.grid(alpha=0.3)
        ax[1].axhline(0, color="grey", lw=0.5)
        ax[1].set_ylim(-1.5, 1.5)
        ax[0].set_ylabel("phase current magnitude, A"); ax[0].legend(fontsize=8, loc="upper right")
        ax[1].set_ylabel("imbalance residual, A, minus its\nmedian over the 2 s before (EXPANDED scale)")
        ax[1].legend(fontsize=8, loc="upper right")
        ax[1].set_xlabel(f"s from start-up onset (dotted line: {lab})")
        fig.suptitle(f"Motor {tag}, every sample (120/s)")
        fig.tight_layout(); p = out_dir / f"raw_phases_motor_{tag}_residual_zoom.png"
        fig.savefig(p, dpi=110); plt.close(fig); out.append(p)
    return (stop - on) / fs, out


if __name__ == "__main__":
    print(main(Path(sys.argv[1]), Path(sys.argv[2])))
