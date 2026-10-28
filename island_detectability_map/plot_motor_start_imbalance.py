"""One motor start at raw 120 Hz, phase currents above and imbalance residuals below
(EXPLORATORY figures). Same event and onset as plot_motor_start.py (Thu 2015-10-01
15:06:15 PDT). Writes three figures, one per zoom level. The lower panel uses the same
A-per-division as the upper one, so the cancellation is visible at face value.
Run from analysis/ with PYTHONPATH=. ; needs ../data/upmu/grizzly_head6.npz.

Origin: verification@6984ee68
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from island_detectability_map import phases
from island_detectability_map.run_pulse import block_mean

COL = ["#1f77b4", "#d62728", "#2ca02c"]
NAME = ["A", "B", "C"]
RES = ["A − (B+C)/2", "B − (A+C)/2", "C − (A+B)/2"]


def pick_onset(I, fs):
    I1 = np.vstack([block_mean(r, int(fs)) for r in I])
    ev = phases.step_events(I1, 1.0, win_s=3.0, min_step=1.0, dead_s=30.0)
    ups = sorted([e for e in ev if e["step_sum"] > 0], key=lambda e: -e["step_sum"])[:30]
    S = I.sum(axis=0); best = []
    for e in ups:
        c = int(e["t_s"] * fs)
        w = S[c - int(60 * fs):c + int(60 * fs)]; w = w - w.mean()
        sp = np.abs(np.fft.rfft(w)) ** 2; f = np.fft.rfftfreq(len(w), 1 / fs)
        if sp[(f > 0.97) & (f < 1.03)].sum() / sp[(f > 0.5) & (f < 2.0)].sum() > 0.05:
            continue                                  # site's 1 Hz pulsed load present
        seg = I[:, c - int(5 * fs):c + int(5 * fs)]
        best.append(((seg.max(axis=1) - np.median(seg[:, :int(3 * fs)], axis=1)).mean(), c))
    c = max(best)[1]
    pre = np.median(S[c - int(5 * fs):c - int(2 * fs)])
    ipk = c - int(4 * fs) + int(np.argmax(S[c - int(4 * fs):c + int(4 * fs)]))
    thr = pre + 0.1 * (S[ipk] - pre); on = ipk
    while S[on - 1] > thr:
        on -= 1
    return on


def main(data_dir: Path, out_dir: Path):
    z = np.load(data_dir / "grizzly_head6.npz"); fs = float(z["fs"])
    n = min(len(z[f"C{k}MAG"]) for k in (1, 2, 3))
    I = np.vstack([z[f"C{k}MAG"][:n].astype(np.float64) for k in (1, 2, 3)])
    on = pick_onset(I, fs)
    t_on = np.datetime64("2015-09-30T17:00:00") + np.timedelta64(int(on / fs * 1000), "ms")
    R = np.vstack([phases.imbalance(I, k) for k in range(3)])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for tag, (lo, hi), ms, desc in (("60s", (-15, 45), 0, "60 s, every sample joined by a line"),
                                    ("3s", (-0.5, 2.5), 2.5, "3 s, every sample marked"),
                                    ("0p6s", (-0.1, 0.5), 5, "0.6 s, every sample marked (8.3 ms apart)")):
        i0, i1 = on + int(lo * fs), on + int(hi * fs)
        tt = (np.arange(i0, i1) - on) / fs
        fig, ax = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
        for k in range(3):
            ax[0].plot(tt, I[k, i0:i1], color=COL[k], lw=0.8, marker="o" if ms else None, ms=ms, label=f"phase {NAME[k]}")
            ax[1].plot(tt, R[k, i0:i1], color=COL[k], lw=0.8, marker="o" if ms else None, ms=ms, label=RES[k])
        span = np.ptp(ax[0].get_ylim()); mid = np.mean([R[:, i0:i1].min(), R[:, i0:i1].max()])
        ax[1].set_ylim(mid - span / 2, mid + span / 2)
        ax[0].set_ylabel("phase current magnitude, A"); ax[0].legend(fontsize=8, loc="upper right")
        ax[1].set_ylabel("imbalance residual, A\n(same A per division as above)"); ax[1].legend(fontsize=8, loc="upper right")
        for a in ax:
            a.grid(alpha=0.3)
        ax[1].set_xlabel(f"s from start-up onset ({str(t_on)[:23].replace('T', ' ')} PDT)")
        fig.suptitle(f"One motor start, raw µPMU at 120 samples/s: {desc}")
        fig.tight_layout()
        p = out_dir / f"raw_phases_motor_start_imbalance_{tag}.png"
        fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)
    return on, paths


if __name__ == "__main__":
    print(main(Path(sys.argv[1]), Path(sys.argv[2])))
