"""One motor start at raw 120 Hz (EXPLORATORY figure). Picks the largest start-up spike among the
biggest 3-s up-steps, skipping windows where the site's 1 Hz pulsed load is present, and times the
onset from the spike itself. Writes /tmp/figs/raw_phases_motor_start_120hz.png (copied to results).
Run from analysis/ with PYTHONPATH=. ; needs ../data/upmu/grizzly_head6.npz.

Origin: verification@4b33e5c0
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from island_detectability_map import phases
from island_detectability_map.run_pulse import block_mean
z = np.load("../data/upmu/grizzly_head6.npz"); fs = float(z["fs"])
n = min(len(z[f"C{k}MAG"]) for k in (1,2,3))
I = np.vstack([z[f"C{k}MAG"][:n].astype(np.float64) for k in (1,2,3)])
I1 = np.vstack([block_mean(r, int(fs)) for r in I])
ev = phases.step_events(I1, 1.0, win_s=3.0, min_step=1.0, dead_s=30.0)
ups = sorted([e for e in ev if e["step_sum"] > 0], key=lambda e: -e["step_sum"])[:30]
best = []
S = I.sum(axis=0)
for e in ups:
    c = int(e["t_s"]*fs)
    w = S[c-int(60*fs):c+int(60*fs)]; w = w - w.mean()
    sp = np.abs(np.fft.rfft(w))**2; f = np.fft.rfftfreq(len(w), 1/fs)
    line = sp[(f>0.97)&(f<1.03)].sum()/sp[(f>0.5)&(f<2.0)].sum()
    if line > 0.05: continue                      # 1 Hz pulsed load present: skip
    seg = I[:, c-int(5*fs):c+int(5*fs)]
    base = np.median(seg[:, :int(3*fs)], axis=1); peak = (seg.max(axis=1)-base).mean()
    best.append((peak, e))
best.sort(key=lambda p: -p[0])
for p, e in best[:5]: print(round(p,2), round(e["t_s"]/3600,3), np.round(e["step"],2))
peak, e = best[0]
c = int(e["t_s"]*fs)
# locate spike onset precisely: first sample where the sum exceeds pre-median + 3 A
s = I.sum(axis=0); pre = np.median(s[c-int(5*fs):c-int(2*fs)])
ipk = c-int(4*fs) + int(np.argmax(s[c-int(4*fs):c+int(4*fs)]))
thr = pre + 0.1*(s[ipk]-pre)
on = ipk
while s[on-1] > thr: on -= 1
T0 = np.datetime64("2015-09-30T17:00:00") + np.timedelta64(int(on/fs*1000),'ms')
print("onset sample", on, "local", T0)
COL = ["#1f77b4", "#d62728", "#2ca02c"]
fig, ax = plt.subplots(3, 1, figsize=(13, 11))
for a, (lo, hi), ms, title in ((ax[0], (-15, 45), 0, "60 s, every sample (120/s) joined by a line"),
                                (ax[1], (-0.5, 2.5), 2.5, "3 s, every sample marked"),
                                (ax[2], (-0.1, 0.5), 5, "0.6 s, every sample marked (8.3 ms apart)")):
    i0, i1 = on + int(lo*fs), on + int(hi*fs)
    tt = (np.arange(i0, i1) - on)/fs
    for k in range(3):
        a.plot(tt, I[k, i0:i1], color=COL[k], lw=0.8, marker="o" if ms else None, ms=ms, label=f"phase {k+1}")
    a.set_title(title, fontsize=10); a.set_ylabel("current magnitude, A"); a.grid(alpha=0.3)
ax[2].set_xlabel(f"s from spike onset ({str(T0)[:23].replace('T',' ')} PDT, 2015)")
ax[0].legend(fontsize=8)
fig.suptitle("One motor start on the LBNL 12 kV bus at the finest resolution the data has (raw µPMU, 120 samples/s)")
fig.tight_layout(); fig.savefig("/tmp/figs/raw_phases_motor_start_120hz.png", dpi=110)
# numbers
k0 = on
pre3 = np.median(I[:, on-int(3*fs):on-int(0.2*fs)], axis=1)
run = np.median(I[:, on+int(3*fs):on+int(10*fs)], axis=1)
pk = I[:, on-5:on+int(1*fs)].max(axis=1); pki = I[:, on-5:on+int(1*fs)].argmax(axis=1)-5
above = (I[:, on:on+int(2*fs)] - pre3[:,None]) > 0.5*(pk-pre3)[:,None]
print("pre", pre3.round(2), "run", run.round(2), "peak", pk.round(2), "peak at samples", pki, "=", (pki/fs*1000).round(0), "ms")
print("samples above half-peak per phase", above.sum(axis=1), "=", (above.sum(axis=1)/fs*1000).round(0), "ms")
print("rise: values first 6 samples from onset-2:", np.round(I[:, on-2:on+6]-pre3[:,None],2))
print("ratio (peak-pre)/(run-pre):", ((pk-pre3)/(run-pre3)).round(2))
