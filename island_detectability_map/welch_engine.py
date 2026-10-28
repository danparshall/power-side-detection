"""Fast, exact equivalent of run_pulse.experiment for a measured (looped) signal trace, plus the
three-channel segment DFTs needed by PNNL-24681's sum statistic.

Why: run_pulse.experiment recomputes a Welch spectrum for every (swing, test hour) pair; with 29
swings x 241 hours x 80 stages that is hours on one core. A Welch bin is |sum of segment DFTs|^2 and
the segment DFT is linear, so the noise and the signal can be transformed once and combined per
swing. The signal is the trace looped from a random start (run_nlr_inject.make_signal_from), so its
segment DFT at bin k and start u is a circular cross-correlation of the trace with the windowed
complex exponential, computed for every start at once by FFT.

Exactness: the random draws are replayed in experiment's order (per swing, per test window: one
uniform for the unused f0, one integer for the start), segments with a non-finite sample are skipped
as pulse.welch_psd skips them, and the baseline, site-line list and statistic follow experiment line
by line. tests/test_follum_pierre.py checks null and injected statistics against experiment itself.
Per-segment mean removal is applied to the signal too (trace_dft_all_starts).

Origin: verification@532a1237
"""
from __future__ import annotations

import numpy as np

from island_detectability_map import run_pulse as R


def _window(L):
    return np.hanning(L)


def segment_dfts(x: np.ndarray, fs: float, win_s: float, seg_s: float, bins: np.ndarray):
    """Per window: (finite-segment DFTs at `bins`, shape segments x bins; segment start offsets).
    Same segmentation, window and mean removal as pulse.welch_psd."""
    n = int(win_s * fs)
    L = int(round(seg_s * fs))
    step = L // 2
    w = _window(L)
    out = []
    for i in range(0, len(x) - n + 1, n):
        xw = x[i:i + n]
        starts = np.arange(0, n - L + 1, step)
        segs = np.stack([xw[s:s + L] for s in starts])
        ok = np.all(np.isfinite(segs), axis=1)
        segs = segs[ok]
        Z = np.fft.rfft((segs - segs.mean(axis=1, keepdims=True)) * w, axis=1)[:, bins]
        out.append((Z, starts[ok]))
    return out


def trace_dft_all_starts(trace: np.ndarray, L: int, bins: np.ndarray) -> np.ndarray:
    """Y[b, u] = sum_n w[n] exp(-2j pi bins[b] n / L) (trace[(u + n) mod T] - segment mean), every start u.
    The segment mean is removed as pulse.welch_psd removes it (np.hanning is the symmetric window, so
    the mean leaks a little into every bin)."""
    T = len(trace)
    w = _window(L)
    n = np.arange(L)
    Ft = np.fft.fft(trace)
    xcorr = lambda h: np.fft.ifft(Ft * np.conj(np.fft.fft(np.conj(h))))   # sum_n h[n] trace[u + n]
    box = np.zeros(T); box[:L] = 1.0 / L
    seg_mean = np.real(xcorr(box))
    Y = np.empty((len(bins), T), complex)
    for b, k in enumerate(bins):
        h = np.zeros(T, complex)
        h[:L] = w * np.exp(-2j * np.pi * k * n / L)
        Y[b] = xcorr(h) - seg_mean * np.sum(h[:L])
    return Y


def _psd_scale(L, fs):
    w = _window(L)
    return 2.0 / (fs * np.sum(w * w))          # one-sided, interior bins


def experiment(x, fs, win_s, trace, seed, swings, char_idx, char_hours=36.0, notch_ratio=10.0, notch_pad=2,
               whitelist=(), band=R.BAND, f0_range=R.F0_RANGE, seg_s=100.0):
    """run_pulse.experiment(..., make_signal=make_signal_from(trace), return_null=True, return_stats=True)
    with rng = default_rng(seed). Returns null_stats, inj_stats (swings x test windows), notched_bins,
    band_bins, notched_freqs_hz, plus: keep (bool over rfft bins), baseline (median over char_idx, all
    band bins), ratios (test windows x kept bins), segments (per window), test (window indices)."""
    st = prepare(x, fs, win_s, trace, seed, len(swings), char_idx, char_hours, notch_ratio, notch_pad,
                 whitelist, band, f0_range, seg_s)
    null, inj = inject(st, swings)
    return {"null_stats": [float(v) for v in null], "inj_stats": inj, "notched_bins": int(st["notch"].sum()),
            "band_bins": int(st["sel"].sum()), "notched_freqs_hz": [round(float(v), 2) for v in st["freqs"][st["noisy"]]],
            "keep": st["keep"], "baseline": st["baseline"], "ratios": st["ratios"],
            "segments": np.array([len(o) for _, o in st["D"]]), "test": st["test"], "freqs": st["freqs"]}


def prepare(x, fs, win_s, trace, seed, n_swings, char_idx, char_hours=36.0, notch_ratio=10.0, notch_pad=2,
            whitelist=(), band=R.BAND, f0_range=R.F0_RANGE, seg_s=100.0):
    """Everything `experiment` computes before it scales the signal: noise segment DFTs, baseline,
    site-line notches, the no-signal ratios, the signal's segment DFTs and the random starts (drawn for
    n_swings swings, as experiment draws them). `inject` then scores any list of n_swings swings, so a
    channel that carries the job times a coefficient c is scored exactly with swings c * scales."""
    n = int(win_s * fs)
    L = int(round(seg_s * fs))
    f = np.fft.rfftfreq(L, 1.0 / fs)
    sel = (f >= band[0]) & (f <= band[1])
    sb = np.flatnonzero(sel)
    D = segment_dfts(x, fs, win_s, seg_s, sb)
    sc = _psd_scale(L, fs)
    spectra = np.zeros((len(D), len(f)))
    for i, (Z, _) in enumerate(D):
        spectra[i, sb] = np.mean(np.abs(Z) ** 2, axis=0) * sc
    n_char = int(char_hours * 3600 / win_s)
    cidx = list(char_idx)
    baseline = np.median(spectra[cidx], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        noisy = (spectra[cidx].max(axis=0) / baseline >= notch_ratio) & sel
    notch = noisy.copy()
    for k in range(1, notch_pad + 1):
        notch[k:] |= noisy[:-k]; notch[:-k] |= noisy[k:]
    for w0 in whitelist:
        notch |= (np.abs(f - w0) <= max(R.RESOLVED_HALF_WIDTH_HZ, 2.0 / seg_s)) & sel
    keep = sel & ~notch
    test = list(range(n_char, len(D)))
    ratios = spectra[test][:, keep] / baseline[keep]
    # signal: same draws as experiment (per swing, per test window: f0 uniform, then start integer)
    rng = np.random.default_rng(seed)
    starts = np.empty((n_swings, len(test)), int)
    for j in range(n_swings):
        for t in range(len(test)):
            rng.uniform(np.log(f0_range[0]), np.log(f0_range[1]))
            starts[j, t] = int(rng.integers(0, len(trace)))
    kb = np.flatnonzero(keep[sb])                 # kept bins, as positions inside sb
    Y = trace_dft_all_starts(np.asarray(trace, float), L, sb[kb])
    return {"D": D, "kb": kb, "Y": Y, "T": len(trace), "starts": starts, "sc": sc, "base_k": baseline[keep],
            "ratios": ratios, "test": test, "keep": keep, "baseline": baseline, "notch": notch, "noisy": noisy,
            "sel": sel, "freqs": f}


def inject(st, swings, reduce=None):
    """(null statistic per test window, injected statistic swings x test windows) for a prepared state.
    reduce maps ratios (rows x kept bins) to one statistic per row; default the maximum over bins, which
    is experiment's statistic."""
    if len(swings) != st["starts"].shape[0]:
        raise ValueError(f"prepared for {st['starts'].shape[0]} swings, got {len(swings)}")
    D, kb, Y, T, starts, sc, base_k = (st[k] for k in ("D", "kb", "Y", "T", "starts", "sc", "base_k"))
    null = st["ratios"].max(axis=1) if reduce is None else reduce(st["ratios"])
    inj = np.empty((len(swings), len(st["test"])))
    for t, i in enumerate(st["test"]):
        Z, offs = D[i]
        Zk = Z[:, kb]                                                   # segments x kept bins
        if reduce is None:
            for j, s in enumerate(swings):
                idx = (starts[j, t] + offs) % T                         # each segment's start in the trace
                Zs = Y[:, idx].T                                        # segments x kept bins
                p = np.mean(np.abs(Zk + s * Zs) ** 2, axis=0) * sc
                inj[j, t] = float(np.max(p / base_k))
        else:
            P = np.stack([np.mean(np.abs(Zk + s * Y[:, (starts[j, t] + offs) % T].T) ** 2, axis=0) * sc
                          for j, s in enumerate(swings)])
            inj[:, t] = reduce(P / base_k)
    return null, inj


def sum_statistic(X, fs, win_s, trace, seed, swings, char_idx, coefs, p_fas, char_hours=36.0, notch_ratio=10.0,
                  notch_pad=2, whitelist=(), band=R.BAND, f0_range=R.F0_RANGE, seg_s=100.0):
    """PNNL-24681 multichannel statistic T(f) = sum_m nu_h P_m(f) / Phi_m(f) on channels X (M x n), offline
    variant: Phi_m = median of the char_idx windows / median of chi2(nu)/nu; site lines notched per channel
    as in `experiment` and pooled (a bin notched on any channel is not searched); GMSC from the learning
    windows' averaged cross-spectral matrix. A segment is used only if every channel is finite in it.
    Injection: the trace (same draws as `experiment`) enters channel m with coefficient coefs[k][m] for
    placement k (for the three unbalance residuals and a load on phase k: +1 on k, -1/2 on the others).
    Returns, per threshold mode ("gmsc", "independent", "identical") and target P_FA, the hourly maximum of
    T(f)/gamma(f) (an alarm when > 1) for the null and for every placement x swing x test window; also
    the raw hourly max of T, G per kept bin, nu per window."""
    from island_detectability_map import follum_pierre as FP
    X = np.asarray(X, float)
    X = X + 0.0 * X.sum(axis=0)                  # NaN in any channel -> NaN in all
    M = X.shape[0]
    L = int(round(seg_s * fs))
    f = np.fft.rfftfreq(L, 1.0 / fs)
    sel = (f >= band[0]) & (f <= band[1])
    sb = np.flatnonzero(sel)
    D = [segment_dfts(X[m], fs, win_s, seg_s, sb) for m in range(M)]
    sc = _psd_scale(L, fs)
    nw = len(D[0])
    spectra = np.zeros((M, nw, len(f)))
    for m in range(M):
        for i, (Z, _) in enumerate(D[m]):
            spectra[m, i, sb] = np.mean(np.abs(Z) ** 2, axis=0) * sc
    cidx = list(char_idx)
    notch = np.zeros(len(f), bool)
    for m in range(M):
        base = np.median(spectra[m][cidx], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            noisy = (spectra[m][cidx].max(axis=0) / base >= notch_ratio) & sel
        nm = noisy.copy()
        for k in range(1, notch_pad + 1):
            nm[k:] |= noisy[:-k]; nm[:-k] |= noisy[k:]
        notch |= nm
    for w0 in whitelist:
        notch |= (np.abs(f - w0) <= max(R.RESOLVED_HALF_WIDTH_HZ, 2.0 / seg_s)) & sel
    keep = sel & ~notch
    kb = np.flatnonzero(keep[sb])
    B = len(kb)
    nseg = np.array([len(o) for _, o in D[0]])
    nu = np.array([FP.effective_dof(L, L // 2, k) for k in nseg])
    nu_c = float(np.median(nu[cidx]))
    amb = np.array([np.median(spectra[m][cidx][:, keep], axis=0) / FP.median_factor(nu_c) for m in range(M)])
    S = np.zeros((B, M, M), complex)
    for i in cidx:
        Zs = np.stack([D[m][i][0][:, kb] for m in range(M)])        # M x segments x B
        S += np.einsum("isb,jsb->bij", Zs, Zs.conj()) / Zs.shape[1]
    G = FP.gmsc(S / len(cidx))
    n_char = int(char_hours * 3600 / win_s)
    test = list(range(n_char, nw))
    modes = {"gmsc": G, "independent": np.zeros(B), "identical": np.ones(B)}
    gam = {(mo, p): np.array([FP.gmsc_threshold(p, B, nu[i], M, g) for i in test]) for mo, g in modes.items() for p in p_fas}

    def T_of(P, t):                              # P: M x B one-sided PSDs of test window t
        return nu[test[t]] * np.sum(P / amb, axis=0)

    null_T = np.empty((len(test), B))
    for t, i in enumerate(test):
        null_T[t] = T_of(np.stack([spectra[m, i][keep] for m in range(M)]), t)
    rng = np.random.default_rng(seed)
    starts = np.empty((len(swings), len(test)), int)
    for j in range(len(swings)):
        for t in range(len(test)):
            rng.uniform(np.log(f0_range[0]), np.log(f0_range[1]))
            starts[j, t] = int(rng.integers(0, len(trace)))
    Y = trace_dft_all_starts(np.asarray(trace, float), L, sb[kb])
    Tl = len(trace)
    coefs = np.asarray(coefs, float)
    inj_T = np.empty((len(coefs), len(swings), len(test), B))
    for t, i in enumerate(test):
        Zn = np.stack([D[m][i][0][:, kb] for m in range(M)])        # M x segs x B
        offs = D[0][i][1]
        for j, s in enumerate(swings):
            Zs = Y[:, (starts[j, t] + offs) % Tl].T                  # segs x B
            for k, c in enumerate(coefs):
                P = np.mean(np.abs(Zn + s * c[:, None, None] * Zs[None]) ** 2, axis=1) * sc
                inj_T[k, j, t] = T_of(P, t)
    out = {"null_Tmax": null_T.max(axis=1), "inj_Tmax": inj_T.max(axis=3), "G": G, "nu": nu, "nu_char": nu_c,
           "kept_bins": B, "band_bins": int(sel.sum()), "test": test, "keep": keep, "amb": amb, "freqs": f, "norm": {}}
    for (mo, p), g in gam.items():
        out["norm"][(mo, p)] = {"null": (null_T / g).max(axis=1), "inj": (inj_T / g[None, None]).max(axis=3)}
    return out
