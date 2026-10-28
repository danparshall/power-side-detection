# power-side-detection

Code behind the power-side detection report (Canary Institute, 2026): can a small AI training
island (16–32 GPUs) be seen from the electricity grid? The method is to add measured GPU training
power traces, by arithmetic, to measured grid records (a 12 kV campus bus read 120 times a second,
and campus meters from 252 kVA to 5.5 MVA read once a second), then score detectors on the result
with thresholds learned on hours they are not scored on.

Nothing here is a field test: these are real training traces added to real grid data.

## Data

No data is stored in this repo. Each source is fetched from its publisher by a script that
records the URL, size and SHA-256 of what it read. Fetched data goes under `data/` (git-ignored).

| Data | Source | Fetch |
|---|---|---|
| A 16-H100 Llama-2-70B LoRA fine-tune (MLPerf), GPU power logs | Vercellino et al. 2026, NLR (formerly NREL) generative-AI power profiles: [arXiv 2604.07345](https://arxiv.org/abs/2604.07345), data [DOI 10.7799/3025227](https://doi.org/10.7799/3025227) | `fetch_nlr.py` |
| 8-GPU H100 and B200 node traces (diffusion, Llama 1B–8B) | Elsayed, Al-Obaidi & Farag 2026, *Scientific Data*, [doi:10.1038/s41597-026-07496-6](https://doi.org/10.1038/s41597-026-07496-6); data [doi:10.6084/m9.figshare.31654879](https://doi.org/10.6084/m9.figshare.31654879) | download the figshare archive by hand; read with `elsayed.py` |
| LBNL 12 kV bus, micro-PMU, 120 samples a second | LBNL Open µPMU data (Stewart et al. 2016), `powerdata-download.lbl.gov` | `fetch_upmu.py` |
| Caltech campus meters, 1-second magnitudes and waveform captures | Caltech SoCal 28-bus dataset, [arXiv 2504.06588](https://arxiv.org/abs/2504.06588), client at [caltech-netlab/digital-twin-dataset](https://github.com/caltech-netlab/digital-twin-dataset) | `caltech_login.py`, then `fetch_caltech.py` |
| UK household one-second mains (for the UK feeder run) | IDEAL homes, Edinburgh DataShare, [DOI 10.7488/ds/2836](https://doi.org/10.7488/ds/2836) (Pullinger et al. 2021) | `fetch_ideal.py` |

Licences differ by source; check each before redistributing anything derived from it. In
particular the Elsayed data is CC BY-NC-ND 4.0 (no derivatives, non-commercial), so this repo ships
the scripts that build the injected traces, not the traces. The Caltech data needs your own API
access through a GitHub login (`caltech_login.py` explains the two-step device flow).

## Install and test

```
uv sync            # or: pip install numpy scipy pandas matplotlib pyyaml requests pytest
uv run pytest      # unit tests; no data needed
```

## Run

Run from the repo root. Outputs go under `results/` (git-ignored).

```
# LBNL bus
python -m island_detectability_map.fetch_upmu data/upmu
python -m island_detectability_map.build_upmu_series data/upmu
python -m island_detectability_map.build_upmu_lowpass data/upmu
python -m island_detectability_map.run_pulse   data/upmu results/20260930_island_detectability_map
python -m island_detectability_map.run_heldout data/upmu results/heldout
python -m island_detectability_map.run         results/20260930_island_detectability_map

# The measured 16-H100 job
python -m island_detectability_map.fetch_nlr data/nlr
python -m island_detectability_map.run_task2 data/upmu data/nlr results/task2
python -m island_detectability_map.run_task3 data/caltech NETWORK_JSON data/upmu data/nlr results/task3

# UK feeders built from IDEAL homes
python -m island_detectability_map.fetch_ideal data/ideal HOME_CSV START END
python -m island_detectability_map.run_uk_feeders data/ideal data/nlr results/uk_feeders

# Synthetic look-alike loads (figures)
python -m synthetic_lookalikes.make_figures results/synthetic_lookalikes
```

Every script states its own inputs, outputs and usage in its docstring. Scripts marked
EXPLORATORY wire tested pieces together and have no tests of their own.

## What is in it

- `island_detectability_map/`: readers for each dataset, the detectors (spectral, edge count,
  notch, change in level), the per-phase and phasor views, held-out thresholds, and the injection
  runs. `params.yaml` tags each number as measured, assumed or taken from a memo.
  - `model.py`: three detection frontiers. A is the energy mean shift on the total alone; B is
    an energy balance against downstream meters; C is the training-step pulse.
  - `pulse.py`: Welch spectrum, a synthetic straggler pulse, and the baseline-ratio line detector.
  - `run_pulse.py`: background spectrum and injection on the LBNL bus. The first 36 hours
    characterise the site; the rest are scored. Seeded, so results are reproducible from the
    same bytes.
- `synthetic_lookalikes/`: one hour of synthetic power at 10 ms for seven loads (EV charger, heat
  pump, laser-printer fuser, well pump, a 16-GPU training job, a crypto miner, an inference
  server), as real power and as the RMS current a 240 V service monitor sees. The training job's
  step (10.2 s, 44% low), level and swing are the logged NLR fine-tune at a 16 kW island; the
  other six are appliance models, not traces. Every number is in `params.yaml`.
- `tests/`: unit tests, mostly written before the code.

## Provenance

This code was developed in the author's private research repository. Each module's docstring
ends with `Origin: verification@<commit>`, the commit that first added it there, and a few
comments cite another commit the same way. If you want the reasoning behind a choice, ask the
author and quote the commit.

## Licence

MIT (see `LICENSE`). The licence covers the code only, not the data it fetches.
