"""Change-in-level (turn-on / turn-off) detector on one-second data, keeping the DC component.

Statistic, per comparison length k (s): S_k(tau) = mean(x[tau+1 : tau+1+k]) - mean(x[tau-k : tau]) (the
transition second skipped, as run_sync.upsteps), on the grid tau_j = j k/2 wherever both spans lie in the
record. A side with fewer than 90% finite samples makes S_k NaN; otherwise each side is the mean of its
finite samples. z_k = S_k / sigma_k, sigma_k = 1.4826 x MAD of S_k over the learning points. Up alarms
(turn-on) where z_k exceeds the worst learning-point z_k x margin; down alarms (turn-off) the same on -z_k.
For a family of channels (the three unbalance residuals) the statistic is the maximum over channels,
each normalized by its own sigma, with one threshold per k and direction.

A step at an arbitrary time lies within k/4 of a grid point, so the best grid point sees at least 0.75 of
its height (the difference falls off as a triangle, h (1 - delta/k)).

Origin: verification@4019f777
"""
from __future__ import annotations

import numpy as np

K = (120, 300, 600, 1200, 3600)
MAD_TO_SIGMA = 1.4826
MIN_FINITE = 0.9


def grid(n, k):
    """Candidate times j k/2 with [tau - k, tau) and [tau + 1, tau + 1 + k) inside [0, n)."""
    return np.arange(k, n - k, k // 2)


def steps_at(x, k, taus):
    x = np.asarray(x, float)
    taus = np.asarray(taus, int)
    fin = np.isfinite(x)
    c = np.concatenate([[0.0], np.cumsum(np.where(fin, x, 0.0))])
    m = np.concatenate([[0], np.cumsum(fin)])
    nb = m[taus] - m[taus - k]
    na = m[taus + 1 + k] - m[taus + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        S = (c[taus + 1 + k] - c[taus + 1]) / na - (c[taus] - c[taus - k]) / nb
    S[(nb < MIN_FINITE * k) | (na < MIN_FINITE * k)] = np.nan
    return S


def steps(x, k):
    taus = grid(len(x), k)
    return taus, steps_at(x, k, taus)


def span_ok(taus, k, hours, hour_n=3600):
    """Both spans of each candidate time lie wholly inside the given clock hours."""
    taus = np.asarray(taus, int)
    if taus.size == 0:
        return np.zeros(0, bool)
    h0 = (taus - k) // hour_n
    h1 = (taus + k) // hour_n
    top = int(h1.max()) + 1
    good = np.zeros(top + 1, int)
    hs = [h for h in hours if 0 <= h <= top]
    good[hs] = 1
    cum = np.concatenate([[0], np.cumsum(good)])
    ok = h0 >= 0
    h0c = np.clip(h0, 0, top)
    return ok & (cum[h1 + 1] - cum[h0c] == h1 - h0 + 1)


def family(S_by_channel, sigma):
    """(up, down): the maximum over channels of S_c / sigma_c and of -S_c / sigma_c."""
    z = np.vstack([np.asarray(S, float) / s for S, s in zip(S_by_channel, sigma)])
    return np.fmax.reduce(z, axis=0), np.fmax.reduce(-z, axis=0)


def calibrate(S_by_channel, learn):
    """sigma per channel and the margin-1 thresholds (worst learning-point family statistic per direction)."""
    learn = np.asarray(learn, bool)
    sigma = []
    for S in S_by_channel:
        v = np.asarray(S, float)[learn]
        v = v[np.isfinite(v)]
        sigma.append(float(MAD_TO_SIGMA * np.median(np.abs(v - np.median(v)))))
    up, down = family(S_by_channel, sigma)
    return {"sigma": sigma, "up": float(np.nanmax(up[learn])), "down": float(np.nanmax(down[learn]))}


def alarms(up, down, cal, margin):
    """(up alarms, down alarms): the family statistic strictly above the margin-1 threshold x margin; NaN never alarms."""
    au = np.nan_to_num(np.asarray(up, float), nan=-np.inf) > cal["up"] * margin
    ad = np.nan_to_num(np.asarray(down, float), nan=-np.inf) > cal["down"] * margin
    return au, ad


def job_steps(x, k, taus, job):
    """S_k at taus of the job alone, over x's finite samples (so null + coef * this = S_k of x + coef job)."""
    taus = np.asarray(taus, int)
    if taus.size == 0:
        return np.zeros(0)
    lo, hi = int(taus.min()) - k, int(taus.max()) + 1 + k
    t = np.arange(lo, hi)
    j = np.where(np.isfinite(np.asarray(x, float)[lo:hi]), job(t), np.nan)
    fin = np.isfinite(j)
    c = np.concatenate([[0.0], np.cumsum(np.where(fin, j, 0.0))])
    m = np.concatenate([[0], np.cumsum(fin)])
    r = taus - lo
    nb = m[r] - m[r - k]
    na = m[r + 1 + k] - m[r + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        S = (c[r + 1 + k] - c[r + 1]) / na - (c[r] - c[r - k]) / nb
    S[(nb < MIN_FINITE * k) | (na < MIN_FINITE * k)] = np.nan
    return S


def injected_steps(x, k, taus, job, coef):
    return steps_at(x, k, taus) + coef * job_steps(x, k, taus, job)


def _first(tau, k, ok):
    """(tau, k, raised-at) of the alarm among ok that can be raised first (tau + 1 + k); ties to smaller k."""
    if not ok.any():
        return None
    i = np.flatnonzero(ok)
    raised = tau[i] + 1 + k[i]
    j = i[np.lexsort((k[i], raised))[0]]
    return float(tau[j]), int(k[j]), float(tau[j] + 1 + k[j])


def edge_outcome(tau, k, up, down, t0, dur, tol=60):
    """Turn-on found: an up alarm within tol + k of t0 and no nearer the stop; turn-off: a down alarm within
    tol + k of t0 + dur and no nearer the start. The hit is the first that can be raised; delay is from the
    edge to the end of its after-span; localization is |tau - edge|."""
    tau = np.asarray(tau, float)
    k = np.asarray(k, int)
    stop = t0 + dur
    d_on, d_off = np.abs(tau - t0), np.abs(tau - stop)
    on_ok = np.asarray(up, bool) & (d_on <= tol + k) & (d_on <= d_off)
    off_ok = np.asarray(down, bool) & (d_off <= tol + k) & (d_off <= d_on)
    out = {"on": bool(on_ok.any()), "off": bool(off_ok.any())}
    out["either"] = out["on"] or out["off"]
    for name, ok, edge in (("on", on_ok, t0), ("off", off_ok, stop)):
        h = _first(tau, k, ok)
        out[f"{name}_hit"] = None if h is None else (h[0], h[1])
        out[f"{name}_delay_s"] = float("nan") if h is None else h[2] - edge
        out[f"{name}_loc_s"] = float("nan") if h is None else abs(h[0] - edge)
    return out


def events(times, merge_s=3600):
    """Alarms merged into events: a new event at an alarm merge_s or more after the current event's first
    alarm (run_sync.events' rule)."""
    n, last = 0, -np.inf
    for t in np.sort(np.asarray(times, float)):
        if t - last >= merge_s:
            n += 1
            last = t
    return n


def events_per_day(times, days, merge_s=3600):
    return events(times, merge_s) / days
