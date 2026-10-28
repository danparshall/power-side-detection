"""Figure: what the three-phase sum, one phase and the imbalance residual look like at one
Caltech meter, as measured and with a synthetic single-phase pause-and-restart added.
EXPLORATORY plotting script, no tests. Convo 20261002_caltech_login_and_hour_of_day_split.

Usage: python -m island_detectability_map.plot_residual_traces CALTECH_ROOT NETWORK_JSON METER HOUR_START_UTC OUT_PNG [DEPTH_KVA]

Origin: verification@3091a6f6
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from island_detectability_map import caltech, phases, run_sync as S


def main(root, net, meter, start, out_png, depth=20.0):
    t0 = pd.Timestamp(start); t1 = t0 + pd.Timedelta(hours=1)
    d = caltech.load_element(Path(root) / meter / "magnitudes", caltech.meter_map(net)[meter], t0.isoformat(), t1.isoformat())
    kva = d.kva
    minutes = np.arange(kva.shape[1]) / 60.0
    def series(k):
        return {"Three-phase sum": k.sum(axis=0), "Phase a alone": k[0], "Residual a (phase a minus the mean of b and c)": phases.imbalance(k, 0)}
    raw = series(kva)
    inj = kva.copy()
    a = 25 * 60; b = a + int(S.PAUSE_S)                       # pause on phase a only, minutes 25.0-27.9
    inj[0, a:b] -= depth
    injected = series(inj)
    local = (t0.tz_localize("UTC").tz_convert("America/Los_Angeles")).strftime("%a %Y-%m-%d %H:%M Pacific")
    fig, ax = plt.subplots(4, 2, figsize=(13.5, 11.5), sharex="col")
    colors = ["#444444", "#1f77b4", "#d62728"]
    span = 0.0
    for name in raw:
        x = raw[name] - np.nanmean(raw[name]); span = max(span, np.nanpercentile(np.abs(x), 99.9))
    span = 1.15 * max(span, depth)
    for r, name in enumerate(raw):
        x = raw[name]; m = np.nanmean(x)
        ax[r, 0].plot(minutes, x - m, lw=0.5, color=colors[r])
        ax[r, 0].set_ylim(-span, span)
        ax[r, 0].set_title(f"{name}: mean {m:,.0f} kVA, RMS 1-s step {np.sqrt(np.nanmean(np.diff(x) ** 2)):.1f} kVA", fontsize=10, loc="left")
        ax[r, 0].set_ylabel("kVA, minus the hour's mean")
        y = injected[name]; z = slice(20 * 60, 33 * 60)
        ax[r, 1].plot(minutes[z], y[z] - m, lw=0.7, color=colors[r])
        ax[r, 1].axvspan(a / 60, b / 60, color="gold", alpha=0.25, lw=0)
        ax[r, 1].set_ylim(-span, span)
        ax[r, 1].set_title(f"{name.split(' (')[0]}, with a {depth:g} kVA pause added on phase a", fontsize=10, loc="left")
    # bottom row: the three residuals on their own, finer scale
    z = slice(20 * 60, 33 * 60)
    for k, (p, col) in enumerate(zip("abc", ["#d62728", "#2ca02c", "#9467bd"])):
        x = phases.imbalance(kva, k); m = np.nanmean(x)
        ax[3, 0].plot(minutes, x - m, lw=0.5, color=col, label=f"residual {p}")
        y = phases.imbalance(inj, k)
        ax[3, 1].plot(minutes[z], y[z] - m, lw=0.8, color=col, label=f"residual {p}")
    fine = 1.4 * depth
    ax[3, 0].set_ylim(-fine, fine); ax[3, 1].set_ylim(-fine, fine)
    ax[3, 1].axvspan(a / 60, b / 60, color="gold", alpha=0.25, lw=0)
    ax[3, 0].set_title("All three residuals on a finer scale", fontsize=10, loc="left")
    ax[3, 1].set_title("All three residuals with the pause on phase a (b and c move half as far, the other way)", fontsize=10, loc="left")
    ax[3, 0].set_ylabel("kVA, minus the hour's mean"); ax[3, 0].legend(fontsize=8, ncol=3, loc="upper right"); ax[3, 1].legend(fontsize=8, ncol=3, loc="upper right")
    for axx in ax.ravel():
        axx.axhline(0, color="k", lw=0.3); axx.grid(alpha=0.25)
    ax[3, 0].set_xlabel("minutes into the hour"); ax[3, 1].set_xlabel("minutes into the hour (zoom)")
    fig.suptitle(f"Caltech {meter} (16.5 kV line, about 5.5 MVA), {local}. 1-second data. Top three rows share one vertical scale; the bottom row is finer.\n"
                 f"Left: as measured. Right: a synthetic single-phase island pausing for {S.PAUSE_S:g} s and restarting (shaded).", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130)
    return {k: float(np.sqrt(np.nanmean(np.diff(v) ** 2))) for k, v in raw.items()}, int((~d.mask).sum())


if __name__ == "__main__":
    a = sys.argv
    print(main(a[1], a[2], a[3], a[4], a[5], float(a[6]) if len(a) > 6 else 20.0))
