"""Detection frontiers for a covert persistent load P (kW) hidden behind an
aggregation point carrying mean load M (kW).

Three methods, deliberately kept separate because they need different data:

A. mean_shift  - energy mean shift on the aggregate alone (no downstream meters).
                 Calibrated to the MEASURED S3 ROC (detection of a persistent added load
                 on summed household and zone-substation records; origin: verification@f7a9006e):
                 d' ~= k * P / sigma_fixed(M), k ~ 1 at a 30-day horizon.
B. balance     - energy balance against downstream meter sums (needs host meter data):
                 P_min = 3 * e * M, e = systematic error (floor memo definition).
C. pulse       - narrowband detection of the training iteration pulse at sub-second
                 sampling (see pulse.py). Only for loads that are not smoothed.

All anchor numbers are measured values quoted from branch results; nothing here is
a fitted "truth" - the curves are interpolation aids between measured points.

Origin: verification@f7a9006e
"""
from __future__ import annotations

import numpy as np
from statistics import NormalDist

_N = NormalDist()

# --- measured anchors: hourly residual sigma under a fixed-climatology baseline (S3) ---
# (mean load kW, hourly fixed-climatology residual sigma kW)
RESIDENTIAL_SYNTHETIC = [(2.6, 0.99), (13.7, 3.34), (57.5, 11.9), (278.2, 54.0)]  # summed GoiEner premises
REAL_MIXED = [(3200.0, 980.0), (12100.0, 3360.0)]  # Ausgrid zone substations: 1-5 MW subset, all 200

# --- measured ROC points (S3), 30-day horizon ---
# (label, mean kW, sigma_fixed kW, load kW, AUC)
S3_ROC = [
    ("synthetic N=200, 12 kW", 57.5, 11.9, 12.0, 0.84),
    ("synthetic N=1000, 12 kW", 278.2, 54.0, 12.0, 0.58),
    ("synthetic N=1000, 100 kW", 278.2, 54.0, 100.0, 0.91),
    ("Ausgrid 1-5 MW, 100 kW", 3200.0, 980.0, 100.0, 0.52),
]


def dprime_from_auc(auc: float) -> float:
    """Equal-variance Gaussian: AUC = Phi(d'/sqrt(2))."""
    return float(np.sqrt(2.0) * _N.inv_cdf(auc))


def dprime_target(tpr: float = 0.90, fpr: float = 0.01) -> float:
    return float(_N.inv_cdf(1.0 - fpr) + _N.inv_cdf(tpr))


def _powerlaw(points):
    x = np.log([p[0] for p in points])
    y = np.log([p[1] for p in points])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


_FITS = {"residential_synthetic": _powerlaw(RESIDENTIAL_SYNTHETIC), "real_mixed": _powerlaw(REAL_MIXED)}


def sigma_exponent(curve: str) -> float:
    return _FITS[curve][0]


def sigma_fixed(mean_kw: float, curve: str) -> float:
    """Hourly fixed-climatology residual sigma (kW) at mean load mean_kw.

    'residential_synthetic' is optimistic for the inspector (pure-residential sums
    understate real mixed-load noise). 'real_mixed' is measured only at 3.2 and
    12.1 MW; below ~1 MW it is an EXTRAPOLATION (the unsampled gap noted in S3).
    """
    slope, intercept = _FITS[curve]
    return float(np.exp(intercept + slope * np.log(mean_kw)))


def s3_calibration_points():
    """k = d'(measured AUC) / (P / sigma_fixed) at each usable S3 ROC point."""
    out = []
    for label, mean_kw, sigma_kw, load_kw, auc in S3_ROC:
        d = dprime_from_auc(auc)
        out.append({"label": label, "mean_kw": mean_kw, "load_kw": load_kw, "auc": auc,
                    "dprime": d, "k": d / (load_kw / sigma_kw)})
    return out


def mean_shift_min_load(mean_kw: float, curve: str, tpr: float = 0.90, fpr: float = 0.01, k: float = 1.2) -> float:
    """Smallest persistent load (kW) detected at (tpr, fpr) by the S3 detector, 30-day horizon."""
    return dprime_target(tpr, fpr) / k * sigma_fixed(mean_kw, curve)


def balance_min_load(mean_kw: float, e: float, kfac: float = 3.0) -> float:
    """Floor-memo definition: detection floor = kfac * e * throughput."""
    return kfac * e * mean_kw


def chips(load_kw: float, kw_per_chip: float = 1.7) -> float:
    """H100-class accelerators for a grid-side load (floor memo: ~1.7 kW grid each)."""
    return load_kw / kw_per_chip


def pulse_min_load(mean_kw: float, depth: float, beta: float, t_hours: float,
                   swing_ref_kva: float, m_ref_kw: float, t_ref_h: float) -> float:
    """Smallest UNSMOOTHED island (kW) whose iteration pulse is detected.

    swing_ref_kva is the MEASURED detectable peak-to-peak swing at aggregation m_ref_kw
    with t_ref_h of data (pulse_measurement.json). Two scalings are ASSUMED, not measured:
    background amplitude ~ M^beta (0.5 = independent loads, 1 = fully correlated), and
    detectable amplitude ~ T^(-1/4) (incoherent averaging of spectra).
    `depth` = swing / island power (modulation depth of the training pulse).
    """
    swing = swing_ref_kva * (mean_kw / m_ref_kw) ** beta * (t_hours / t_ref_h) ** -0.25
    return swing / depth


def max_aggregation(frontier, load_kw: float, lo: float = 1.0, hi: float = 1e7) -> float:
    """Largest mean load M (kW) at which frontier(M) <= load_kw (frontier increasing in M)."""
    if frontier(lo) > load_kw:
        return float("nan")
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if frontier(mid) <= load_kw:
            lo = mid
        else:
            hi = mid
    return float(lo)
