"""Regenerate the look-alike figures: python -m synthetic_lookalikes.make_figures OUT_DIR

Origin: verification@81ea80da
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from synthetic_lookalikes import PARAMS, currents, loads, spectrogram

LABELS = {
    "ev_charger": "EV charger (level 2)",
    "heat_pump": "Heat pump (thermostatic)",
    "laser_fuser": "Laser printer fuser",
    "well_pump": "Well pump (intermittent)",
    "training_16gpu": "Training, 16 GPUs (10.2 s step)",
    "crypto_miner": "Crypto miner (flat)",
    "inference_server": "Inference server (bursty)",
}
INK = "#3d3d3a"
ORDER = ["training_16gpu", "ev_charger", "heat_pump", "laser_fuser", "well_pump", "crypto_miner",
         "inference_server"]   # display order: the training job first
ZOOM_START_S = {  # a minute in which each load is doing something
    "ev_charger": 295, "heat_pump": 895, "laser_fuser": 595, "well_pump": 835,
    "training_16gpu": 1500, "crypto_miner": 1950, "inference_server": 1500,
}


def time_series(P, out, L, unit, fname, title):
    L = {k: L[k] for k in ORDER}
    t = np.arange(int(P["duration_s"] / P["dt_s"])) * P["dt_s"]
    h = 1.6 * len(L)
    fig, axes = plt.subplots(len(L), 2, figsize=(11, h), sharex=False,
                             gridspec_kw={"width_ratios": [2, 1], "wspace": 0.14, "hspace": 0.45})
    fig.subplots_adjust(left=0.07, right=0.99, top=1 - 0.85 / h, bottom=0.4 / h)
    for (name, p), (ax_full, ax_zoom) in zip(L.items(), axes):
        ax_full.plot(t / 60, p, lw=0.6, color=INK)
        z0 = ZOOM_START_S[name]
        z = (t >= z0) & (t < z0 + 60)
        ax_zoom.plot(t[z] - z0, p[z], lw=0.8, color=INK)
        ax_zoom.set_title(f"60 s from {z0 / 60:.1f} min", loc="left", fontsize=8, color=INK)
        ax_full.set_title(LABELS[name], loc="left", fontsize=9, color=INK)
        for ax in (ax_full, ax_zoom):
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8, colors=INK)
            ax.grid(axis="y", lw=0.4, alpha=0.3)
        ax_full.set_ylabel(unit, fontsize=8, color=INK)
        if ax_full is not axes[-1, 0]:
            ax_full.tick_params(labelbottom=False)
            ax_zoom.tick_params(labelbottom=False)
    axes[-1, 0].set_xlabel("minutes", fontsize=8, color=INK)
    axes[-1, 1].set_xlabel("seconds", fontsize=8, color=INK)
    fig.suptitle(title, fontsize=10, color=INK, x=0.07, ha="left")
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


SHORT = {
    "ev_charger": "EV charger", "heat_pump": "Heat pump", "laser_fuser": "Laser fuser",
    "well_pump": "Well pump", "training_16gpu": "Training, 16 GPUs", "crypto_miner": "Crypto miner",
    "inference_server": "Inference server",
}


def log_bin(f, S, per_decade=150):
    """Mean PSD in log-spaced frequency bins, for display only."""
    df = f[1] - f[0]
    proposed = np.logspace(np.log10(f[0]), np.log10(f[-1]), int(per_decade * np.log10(f[-1] / f[0])) + 1)
    edges = [proposed[0] - df / 2]
    for e in proposed[1:]:
        if e - edges[-1] >= df:  # never finer than the FFT resolution
            edges.append(e)
    edges = np.array(edges)
    idx = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, len(edges) - 2)
    out = np.zeros((len(edges) - 1, S.shape[1]))
    counts = np.bincount(idx, minlength=len(edges) - 1)
    for j in range(S.shape[1]):
        out[:, j] = np.bincount(idx, weights=S[:, j], minlength=len(edges) - 1) / np.maximum(counts, 1)
    return edges, out


def spectrograms(P, out, L, unit, fname):
    L = {k: L[k] for k in ORDER}
    fig, axes = plt.subplots(1, len(L), figsize=(15, 4.2), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    fig.subplots_adjust(left=0.045, right=0.93, top=0.84, bottom=0.14)
    cols = {}
    for name, p in L.items():
        f, tw, S = spectrogram(p, P)
        edges_f, Sb = log_bin(f, S)
        cols[name] = (edges_f, tw - P["spectrogram"]["window_s"] / 2, 10 * np.log10(Sb + 1e-12))
    allv = np.concatenate([c[2].ravel() for c in cols.values()])
    vmin, vmax = np.percentile(allv, [5, 99.5])
    w = P["spectrogram"]["window_s"]
    for ax, (name, (edges_f, t0, dB)) in zip(axes, cols.items()):
        edges_t = np.concatenate([t0, [t0[-1] + P["spectrogram"]["stride_s"]]]) / 60
        m = ax.pcolormesh(edges_t, edges_f, dB, cmap="Blues", vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_yscale("log")
        ax.set_title(SHORT[name], fontsize=8, color=INK, loc="left")
        ax.tick_params(labelsize=7, colors=INK)
        ax.set_xlabel("window start, min", fontsize=7, color=INK)
    axes[0].set_ylabel("Hz", fontsize=8, color=INK)
    axes[0].set_ylim(P["spectrogram"]["f_min_hz"], 50)
    cb = fig.colorbar(m, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label(f"PSD, dB re 1 {unit}²/Hz (mean removed; log-binned)", fontsize=7, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK)
    fig.suptitle(f"Spectrograms of {unit}: {w // 60}-min Hann window, {P['spectrogram']['stride_s'] // 60}-min stride",
                 fontsize=10, color=INK, x=0.045, ha="left")
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    kw = loads(PARAMS)
    amps = currents(PARAMS)
    v = PARAMS["volts"]
    time_series(PARAMS, out, kw, "kW", "lookalikes_time_series.png",
                "Synthetic look-alike loads, real power, one hour at 10 ms (params.yaml)")
    spectrograms(PARAMS, out, kw, "kW", "lookalikes_spectrograms.png")
    time_series(PARAMS, out, amps, "A", "lookalikes_current_time_series.png",
                f"What a service monitor sees: RMS current at {v} V, motor starts as locked-rotor kVA at pf 0.4")
    spectrograms(PARAMS, out, amps, "A", "lookalikes_current_spectrograms.png")
    print(f"wrote {out}")
