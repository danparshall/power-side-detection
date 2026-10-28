"""Plots of the raw per-phase currents on the LBNL 12 kV bus (grizzly_bus1_2).

EXPLORATORY (figures only; event finding uses the tested phases.step_events).
Question: what do motors switching on and off look like, phase by phase?
Writes raw_phases_{overview,zoom,gallery,details}.png.
Usage: python -m island_detectability_map.plot_raw_phases ../data/upmu OUT_DIR
Units: current magnitude in A per phase; at ~7.4 kV line-to-neutral, 1 A on one phase is ~7.4 kVA.
Times in local Berkeley time (PDT = UTC-7; the record starts 2015-10-01 00:00 UTC).

Origin: verification@05655747
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from island_detectability_map import phases
from island_detectability_map.run_pulse import block_mean

COL = ["#1f77b4", "#d62728", "#2ca02c"]
PH = ["phase 1", "phase 2", "phase 3"]
T0_LOCAL = np.datetime64("2015-09-30T17:00:00")          # 2015-10-01 00:00 UTC in PDT


def to_dt(sec):
    return T0_LOCAL + (np.asarray(sec) * 1e3).astype("timedelta64[ms]")


def imb(x):
    return np.vstack([phases.imbalance(x, k) for k in range(3)])


def main(data_dir: Path, out_dir: Path):
    z = np.load(data_dir / "grizzly_head6.npz")
    n = min(len(z[f"C{k}MAG"]) for k in (1, 2, 3))
    I = np.vstack([z[f"C{k}MAG"][:n].astype(np.float64) for k in (1, 2, 3)])
    fs = float(z["fs"])
    I1 = np.vstack([block_mean(r, int(fs)) for r in I])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. overview, 1-min means
    m = np.vstack([block_mean(r, 60) for r in I1]); tm = to_dt(np.arange(m.shape[1]) * 60 + 30)
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for k in range(3):
        ax[0].plot(tm, m[k], color=COL[k], lw=0.7, label=f"{PH[k]} (mean {I[k].mean():.1f} A)")
    ax[0].set_ylabel("current, A (1-min mean)"); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("LBNL 12 kV bus, raw per-phase current, 76 h from Wed 30 Sep 2015 17:00 PDT")
    r = imb(m)
    for k in range(3):
        ax[1].plot(tm, r[k] - np.median(r[k]), color=COL[k], lw=0.7, label=f"imbalance, {PH[k]}")
    ax[1].set_ylabel("imbalance residual, A\n(minus its median)"); ax[1].legend(loc="upper right", fontsize=8)
    ax[1].set_ylim(-0.5 * np.ptp(ax[0].get_ylim()) / 2, 0.5 * np.ptp(ax[0].get_ylim()) / 2)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    fig.tight_layout(); fig.savefig(out_dir / "raw_phases_overview.png", dpi=110); plt.close(fig)

    # 2. three-hour zoom, 1-s data (morning ramp with many motor starts)
    a, b = int(13.5 * 3600), int(16.5 * 3600)
    t = to_dt(np.arange(a, b) + 0.5)
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for k in range(3):
        ax[0].plot(t, I1[k, a:b], color=COL[k], lw=0.5, label=PH[k])
    ax[0].set_ylabel("current, A (1-s mean)"); ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title("Three hours at 1-s resolution (Thu morning): the spikes are motor starts, balanced across phases")
    r = imb(I1[:, a:b])
    for k in range(3):
        ax[1].plot(t, r[k] - np.median(r[k]), color=COL[k], lw=0.5)
    span = np.ptp(ax[0].get_ylim())
    ax[1].set_ylim(-span / 4, span / 4)
    ax[1].set_ylabel("imbalance residual, A\n(same scale as above, halved)")
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout(); fig.savefig(out_dir / "raw_phases_zoom.png", dpi=110); plt.close(fig)

    # 3. gallery, full 120 Hz, change from the pre-event median. Row 1: the largest events the
    # 3-s step detector finds (they turn out to be start spikes plus short runs). Rows 2-3:
    # sustained switch-ons and switch-offs found with 20-s windows (spikes average out).
    ev = phases.step_events(I1, 1.0, win_s=3.0, min_step=1.0, dead_s=30.0)
    ev20 = phases.step_events(I1, 1.0, win_s=20.0, min_step=3.0, dead_s=120.0)
    rows = [("3-s detector, largest", sorted([e for e in ev if e["step_sum"] > 0], key=lambda e: -e["step_sum"])[:4]),
            ("sustained ON", sorted([e for e in ev20 if e["step_sum"] > 0], key=lambda e: -e["step_sum"])[:4]),
            ("sustained OFF", sorted([e for e in ev20 if e["step_sum"] < 0], key=lambda e: e["step_sum"])[:4])]
    ups = rows[0][1]
    fig, ax = plt.subplots(3, 4, figsize=(15, 10), sharey=True)
    for row, (lab, evs) in enumerate(rows):
        for j, e in enumerate(evs):
            c = int(e["t_s"] * fs); s0, s1 = c - int(90 * fs), c + int(90 * fs)
            seg = I[:, s0:s1]; pre = seg[:, : int(60 * fs)]
            base = np.median(pre, axis=1, keepdims=True)
            tt = (np.arange(s0, s1) - c) / fs
            for k in range(3):
                ax[row, j].plot(tt, seg[k] - base[k], color=COL[k], lw=0.4, label=PH[k])
            rr = phases.imbalance(seg, 0) - np.median(phases.imbalance(pre, 0))
            ax[row, j].plot(tt, rr, color="k", lw=0.4, label="imbalance (ph 1)")
            ax[row, j].set_title(f"{lab}: {str(to_dt(e['t_s']))[5:16].replace('T', ' ')} PDT\n"
                                 f"step {np.round(e['step'], 1)} A, balanced {e['balanced_share']:.3f}", fontsize=8)
            ax[row, j].axhline(0, color="grey", lw=0.3)
            if row == 2:
                ax[row, j].set_xlabel("s from step")
        ax[row, 0].set_ylabel("change in current, A (120 Hz)")
    ax[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Motor events in 76 h, raw 120 Hz, change from the pre-event median; black = imbalance residual")
    fig.tight_layout(); fig.savefig(out_dir / "raw_phases_gallery.png", dpi=110); plt.close(fig)

    # 4. details: inrush close-up, the site's 1 Hz pulsed load, the least-balanced step
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    e = ups[0]; c = int(e["t_s"] * fs); s0, s1 = c - int(4 * fs), c + int(8 * fs)
    seg = I[:, s0:s1]; tt = (np.arange(s0, s1) - c) / fs
    for k in range(3):
        ax[0].plot(tt, seg[k], color=COL[k], lw=0.8, label=PH[k])
    ax[0].set_title("Largest switch-on, 12 s at 120 Hz: start-up transient", fontsize=9)
    ax[0].set_xlabel("s from step"); ax[0].set_ylabel("current, A"); ax[0].legend(fontsize=7)
    # hour with the strongest 1.00 Hz line in the summed current
    x12 = np.vstack([block_mean(r, 10) for r in I]); n_h = int(3600 * 12)
    best, bh = -1.0, 0
    for h in range(x12.shape[1] // n_h):
        sp = np.abs(np.fft.rfft(x12[:, h * n_h:(h + 1) * n_h].sum(axis=0) - x12[:, h * n_h:(h + 1) * n_h].sum(axis=0).mean()))
        p = sp[3600 - 2:3600 + 3].sum()
        if p > best:
            best, bh = p, h
    s0 = int((bh * 3600 + 1800) * fs); s1 = s0 + int(10 * fs); tt = np.arange(s1 - s0) / fs
    for k in range(3):
        ax[1].plot(tt, I[k, s0:s1] - I[k, s0:s1].mean(), color=COL[k], lw=0.7)
    ax[1].plot(tt, phases.imbalance(I[:, s0:s1], 0) - phases.imbalance(I[:, s0:s1], 0).mean(), color="k", lw=0.7, label="imbalance (ph 1)")
    ax[1].set_title(f"Site's own 1.00 Hz pulsed load, 10 s at 120 Hz\n({str(to_dt(bh * 3600 + 1800))[5:16].replace('T', ' ')} PDT; minus 10-s mean)", fontsize=9)
    ax[1].set_xlabel("s"); ax[1].legend(fontsize=7)
    lo = sorted(ev, key=lambda e: e["balanced_share"])[0]
    c = int(lo["t_s"] * fs); s0, s1 = c - int(30 * fs), c + int(30 * fs)
    seg = I[:, s0:s1]; base = np.median(seg[:, : int(20 * fs)], axis=1, keepdims=True); tt = (np.arange(s0, s1) - c) / fs
    for k in range(3):
        ax[2].plot(tt, seg[k] - base[k], color=COL[k], lw=0.5)
    ax[2].set_title(f"Least balanced step of {len(ev)} (≥1 A total): {str(to_dt(lo['t_s']))[5:16].replace('T', ' ')} PDT\n"
                    f"step {np.round(lo['step'], 2)} A, balanced share {lo['balanced_share']:.2f}", fontsize=9)
    ax[2].set_xlabel("s from step"); ax[2].set_ylabel("change in current, A")
    fig.tight_layout(); fig.savefig(out_dir / "raw_phases_details.png", dpi=110); plt.close(fig)
    return {"n_events": len(ev), "site_load_hour": bh}


if __name__ == "__main__":
    print(main(Path(sys.argv[1]), Path(sys.argv[2])))
