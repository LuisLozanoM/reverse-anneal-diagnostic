# locth1 — A calibrated diagnostic for reverse-anneal sampling in programmable quantum annealers

Code and data for the paper:

> Luis Lozano, *"A calibrated diagnostic for reverse-anneal sampling in programmable quantum annealers"*,
> **[arXiv:2605.19381](https://arxiv.org/abs/2605.19381)** (v2, 2026). Currently under peer review at a journal.

**If you use this code or data, please cite the arXiv preprint** (see `CITATION.cff`). A citable Zenodo DOI for this repository will be minted from a tagged release upon journal acceptance. Every quantitative result in the paper regenerates from the raw samplesets deposited here — see `DATA_MAP.md` for the figure-to-data map.

## What this is

A subsystem-level validation protocol for reverse-anneal sampling on programmable quantum annealers. Two observables are computed on a small subsystem read out of a D-Wave processor:

- **Memory order parameter `M`** — the maximum total-variation distance between subsystem marginals prepared from different initial states; `M → 0` means the readout has erased its initialization.
- **Conditional-Boltzmann distance `D_TV`** — the distance between the measured subsystem marginal and a fixed, independently calibrated conditional-Boltzmann reference, used strictly as a *discrepancy detector* (not a thermometer).

Together they separate **relaxed**, **memory-retaining**, and **"wrong-basin"** (relaxed-but-non-thermal) readouts — the last being a failure mode that a memory-only or success-probability metric aliases as success. The protocol is demonstrated across two D-Wave QPU generations (Advantage2 / Zephyr and Advantage_system6.4 / Pegasus), and every reported number regenerates from the deposited raw samplesets.

This is a device-validation / benchmarking-methodology contribution. It makes **no** claim of quantum computational advantage, isolated-system thermalization, or many-body thermometry.

## Hardware

- **Advantage2** (Zephyr, 4579 qubits) — primary QPU.
- **Advantage_system6.4** (Pegasus, 5612 qubits) — cross-architecture replication.
- Reverse anneal with a pause at `s_p = 0.4`, `t_p = 100 µs`.
- Classical baselines and analysis: single M3 Max (CPU/MPS); no GPU cluster required.

## Repository layout

| Path | Contents |
|---|---|
| `src/locth1/` | Analysis library: reverse-anneal driver, observables (`M`, `D_TV`), embedding, classical baselines (Glauber, ED, Lindblad, SVMC), Gibbs comparison |
| `scripts/` | Experiment drivers and figure-generation scripts |
| `data/raw/phase6_gibbs/` | Raw HDF5 samplesets (per-read spins + energies + per-job metadata) for the thermal-marginal diagnostic |
| `data/raw/phase4/` | Subsystem–bath sweep samplesets and per-condition summaries |
| `data/classical/phase5/` | Classical-baseline outputs (exact diagonalization, Glauber) |
| `manuscript/` | LaTeX sources (revtex4-2) and figures |
| `tests/` | Unit tests |
| `pyproject.toml` | Package metadata and pinned dependencies |

## Data

The raw QPU data lives in `data/raw/` as HDF5 files (~37 MB total). Each `*.h5` stores the per-read spin configurations, energies, and per-job attributes (solver identifier, `graph_id`, anneal schedule, calibration epoch, initial state). Subsystem marginals, the calibrated conditional-Boltzmann references, and all `D_TV` / `M` values are computed from these files by the analysis code — nothing in the paper is hand-entered.

D-Wave Leap credentials (`dwave.conf`, API tokens) are **not** included and are not required to reproduce any reported quantity — only to regenerate raw samplesets from scratch.

## Reproducing the analysis

No QPU access is needed to reproduce the derived quantities and figures from the included raw data:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                                  # installs the locth1 package
pip install pytest && pytest tests/               # unit tests
python scripts/figures/generate_all_figures.py    # Figs. 1, 2, 3, 5
python scripts/figures/fig6_thermal_marginal.py   # Fig. 4 (thermal-marginal diagnostic)
python scripts/phase6_beta_sensitivity.py         # SI beta-sensitivity figure + JSON
```

The figure-to-data map is in `DATA_MAP.md`.

## License

Code: Apache-2.0. Data: CC-BY-4.0.

## Citation

Please cite the paper and the Zenodo record (DOI minted at publication).

## Contact

Luis Lozano — `lalozanom@tec.mx` — ORCID 0000-0001-7202-3437.
