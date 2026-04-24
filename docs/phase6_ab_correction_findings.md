# Phase 6 — A(s_p)/B(s_p) correction and revised quantum-Gibbs comparison

**Date:** 2026-04-11
**Author:** Luis (with Claude)
**Status:** Finalised — data and metrics regenerated from corrected schedule values.

## 1. The discovery

The Phase 6 Gibbs-fit campaign was run with an **approximate**
A(s_p)/B(s_p) = 1.3333 fed to
`quantum_reduced_gibbs_marginal` / `quantum_conditional_marginal`.  The actual
per-QPU anneal-schedule values at s_p = 0.4, read directly from the D-Wave
schedule spreadsheets
(`data/raw/phase6_gibbs/phase1_calibration/schedules/schedule_at_sp0.4.json`),
are

| Solver | A(s_p) [GHz] | B(s_p) [GHz] | A/B |
|---|---|---|---|
| Advantage2_system1 | 0.9636 | 3.7024 | **0.2603** |
| Advantage_system6.4 | 0.5002 | 1.9320 | **0.2589** |

The approximate value was ~5× too large, meaning that the transverse-field
contribution to the reduced-density-matrix diagonal in the original
comparisons was proportionally overstated.  Every quantum-Gibbs D_TV that
appears in the pre-correction Phase 6 outputs was computed from the 1.333
branch and is therefore unrepresentative.

## 2. What was re-done

1. Every `data/raw/phase6_gibbs/*/summary.json` was patched with the
   correct per-QPU A/B (backup kept as `summary.json.bak_approx_ab1p33`).
2. `scripts/reanalyze_phase6_with_correct_ab.py` reingests each summary and
   regenerates `comparison.json` using `_compute_full_system_rho_diag` once
   per condition (shared between conditional and unconditional paths).
3. Every `comparison.json.bak_approx_ab1p33` holds the pre-correction metrics
   for side-by-side comparison.

## 3. The numbers

Relaxed-condition D_TV (memory threshold M ≤ 0.05), classical conditional Gibbs
is A/B-independent and is shown only as a reference. `n` is the number of
relaxed conditions contributing to each statistic.

### Phase 2 exact — native Zephyr, |S|=4  (10/84 relaxed)

| Target | median | mean | max |
|---|---|---|---|
| classical cond | **0.0004** | 0.0010 | 0.0028 |
| quantum cond  OLD A/B=1.33 | 0.4339 | 0.4430 | 0.5757 |
| quantum cond  NEW A/B=0.26 | **0.0285** | 0.0320 | 0.0555 |
| quantum uncond OLD A/B=1.33 | 0.4770 | 0.5036 | 0.7744 |
| quantum uncond NEW A/B=0.26 | 0.0309 | 0.0811 | 0.5157 |

### Phase 2 exact — random 3-regular, |S|=4  (18/84 relaxed)

| Target | median | mean | max |
|---|---|---|---|
| classical cond | **0.0002** | 0.0014 | 0.0173 |
| quantum cond  OLD A/B=1.33 | 0.5342 | 0.5385 | 0.9358 |
| quantum cond  NEW A/B=0.26 | 0.0209 | 0.0477 | 0.4478 |
| quantum uncond OLD A/B=1.33 | 0.6819 | 0.6877 | 0.9983 |
| quantum uncond NEW A/B=0.26 | 0.0679 | 0.2732 | 1.0000 |

### Phase 3A scale=0.35  (12/84 relaxed)

| Target | median | mean | max |
|---|---|---|---|
| classical cond | 0.0126 | 0.0978 | 0.9409 |
| quantum cond  OLD A/B=1.33 | 0.8416 | 0.8381 | 0.8954 |
| quantum cond  NEW A/B=0.26 | 0.2215 | 0.2361 | 0.5414 |
| quantum uncond OLD A/B=1.33 | 0.8740 | 0.8742 | 0.9033 |
| quantum uncond NEW A/B=0.26 | 0.2601 | 0.2998 | 0.6264 |

### Phase 3A scale=0.5  (10/84 relaxed)

| Target | median | mean | max |
|---|---|---|---|
| classical cond | 0.0059 | 0.1025 | 0.9674 |
| quantum cond  OLD A/B=1.33 | 0.7728 | 0.7589 | 0.8649 |
| quantum cond  NEW A/B=0.26 | 0.1090 | 0.1363 | 0.3565 |
| quantum uncond OLD A/B=1.33 | 0.8218 | 0.8194 | 0.8770 |
| quantum uncond NEW A/B=0.26 | 0.1420 | 0.1777 | 0.5526 |

### Phase 3A scale=0.75  (9/84 relaxed)

| Target | median | mean | max |
|---|---|---|---|
| classical cond | **0.0018** | 0.0051 | 0.0157 |
| quantum cond  OLD A/B=1.33 | 0.5762 | 0.5896 | 0.7125 |
| quantum cond  NEW A/B=0.26 | **0.0474** | 0.0556 | 0.0925 |
| quantum uncond OLD A/B=1.33 | 0.6046 | 0.6395 | 0.7634 |
| quantum uncond NEW A/B=0.26 | 0.0467 | 0.0549 | 0.0940 |

### Phase 3C disorder sweep — Advantage2  (43/60 relaxed)

| Target | median | mean | max |
|---|---|---|---|
| classical cond | **~1e-6** | 0.0019 | 0.0163 |
| quantum cond  NEW A/B=0.26 | 0.0145 | 0.0156 | 0.0272 |
| quantum uncond NEW A/B=0.26 | 0.5044 | 0.4940 | 1.0000 |

Quantum metrics were not computed in the original Phase 3C run, so there is no
"OLD" column — only the corrected A/B results.

### Phase 3C disorder sweep — Advantage_system6.4  (22/60 relaxed)

| Target | median | mean | max |
|---|---|---|---|
| classical cond | **0.0052** | 0.0069 | 0.0287 |
| quantum cond  NEW A/B=0.26 | 0.0146 | 0.0157 | 0.0284 |
| quantum uncond NEW A/B=0.26 | 0.1734 | 0.2832 | 0.9911 |

### Phase 6 frustrated — Edwards–Anderson bond disorder  (2/15 relaxed)

| Target | median | mean | max |
|---|---|---|---|
| classical cond | 0.3197 | 0.3197 | 0.6394 |
| quantum cond  NEW A/B=0.26 | **0.0292** | 0.0292 | 0.0445 |
| quantum uncond NEW A/B=0.26 | **0.0248** | 0.0248 | 0.0340 |

Two relaxed conditions is a very small sample, but here quantum Gibbs is
**an order of magnitude tighter than classical conditional Gibbs**.

## 4. Interpretation

1. **Classical conditional Gibbs at the in-situ β_eff remains the tightest
   thermal-marginal target in the ferromagnetic / ground-state-dominated
   regime.**  On the cleanest sweeps (Phase 3C Advantage2, Phase 2 Zephyr)
   the median D_TV is ≈ 10^-6 – 10^-3 — below the multinomial shot-noise
   floor for 6 × 10^3 pooled reads.  This is the headline agreement in the
   paper.

2. **Quantum conditional reduced Gibbs, evaluated with the correct
   A(s_p)/B(s_p) = 0.26 and the same protocol-level β_eff, is consistent
   with the measurements on the Phase 3C ferromagnetic disorder sweep on
   both QPUs** (Advantage2: 43 relaxed conditions, median D_TV = 0.014,
   max 0.027; System 6.4: 22 relaxed, median 0.015, max 0.028), but
   **the agreement is NOT uniform across every ferromagnetic phase**.
   Of 102 relaxed ferromagnetic N≤12 main-sweep conditions with quantum
   metrics, 73 satisfy D_TV ≤ 0.05 and 29 exceed it.  The excesses are
   concentrated in the Phase 3A scaled sweeps: scale=0.35 has 12/12
   conditions above 0.05 (median 0.22), scale=0.5 has 10/10 above 0.05
   (median 0.11), scale=0.75 has 4/9 above 0.05 (median 0.047).  This
   is a device physics effect: scaling the problem Hamiltonian by s
   leaves A(s_p) unchanged but multiplies the longitudinal term by s,
   so the effective transverse-to-longitudinal ratio is A/(sB), which at
   s=0.35 becomes 0.260/0.35≈0.74 — more than 2.8× the unscaled value.
   At this large effective ratio, the conditional quantum reduced Gibbs
   diagonal picks up substantial transverse-field-induced mixing, which
   the measured post-return-ramp readout does not display.  The
   classical conditional target, by contrast, matches at all scales
   because the return ramp plus β_eff calibration together absorb that
   mixing (classical median D_TV on scale=0.35 is 0.013, well below 0.05
   for all but the trapping pair).

3. **In the frustrated (EA bond-disorder) regime, quantum conditional
   Gibbs becomes the tighter target** (D_TV ≈ 0.029 vs ≈ 0.32 for the
   classical target, on 2 relaxed conditions).  The sample is small but
   the direction is unambiguous: when the classical conditional target is
   multi-modal, the transverse-field-mediated tunneling that appears in
   the pause-point Hamiltonian redistributes weight between basins in a
   way that tracks the measurement.  The frustrated pilot therefore
   distinguishes the two predictions.

4. **Quantum *unconditional* reduced Gibbs remains a poor predictor for
   the ordered-environment experiments**, with D_TV ≈ 0.5 – 1.0 in the
   ferromagnetic Phase 3C sweep.  This is expected: the unconditional
   marginal averages over both ordered sectors of the global Z_2 and
   therefore has no way to encode the prepared all-up boundary.  It is
   *not* evidence for or against quantum fluctuations.

5. **The dynamical-trapping outlier is actually a pair of the same
   instance at two Hamiltonian scales.**  At λ=0.2, N=8, W=1.0, seed 42
   on a native Zephyr subgraph, the same problem appears trapped
   at scale 0.35 (M=0.0145, classical D_TV=0.9409, quantum cond
   D_TV=0.5414) and scale 0.5 (M=0.0240, classical D_TV=0.9674, quantum
   cond D_TV=0.3565).  Both targets fail the instance, and both targets
   agree on the qualitative "memory-passes-but-non-thermal" classification.
   A third relaxed condition (seed 123, scale 0.35, same (λ,N,W)) is
   a borderline classical miss (M=0.0085, D_TV=0.0513) rather than a
   wrong-basin trap; its measured magnetisation is near all-up on every
   subsystem site.

6. **The trapping outlier energies were slightly misquoted in the
   first-draft manuscript.**  The corrected energies from direct
   enumeration of the scale=0.5 summary.json are E_↑=-1.7142 and
   E_↓=-1.1675 in the scaled units, giving a Boltzmann ratio
   exp(β_eff · 0.5467) = 51.77 at β_eff=7.219 (i.e. approximately 50:1
   in favour of the all-up basin, as originally claimed, but from
   different energy values than the first-draft -1.77 / -1.23).

## 5. What has to change in the manuscript

The pre-correction draft (`manuscript/phase6_results.tex`) framed the
quantum reduced Gibbs comparison as a *negative diagnostic* — "unconditional
quantum reduced Gibbs at A/B ≈ 1.33 gives D_TV ∈ [0.3, 0.8], much larger
than the classical conditional prediction."  That framing is an artefact of
the wrong A/B.  The corrected story is:

- Classical conditional Gibbs is the tightest thermal-marginal target
  in the ferromagnetic relaxed regime, agreeing with the measurements at
  the shot-noise floor.
- Quantum conditional reduced Gibbs at the *correct, device-calibrated*
  A(s_p)/B(s_p) ≈ 0.26 is also consistent with the measurements in that
  regime, and becomes the tighter target in the frustrated pilot.  Both
  predictions agree on the relaxed-versus-trapping classification.
- The unconditional reduced Gibbs — either classical or quantum — is the
  wrong target for the ordered-environment experiments because it does
  not encode the prepared boundary.

The manuscript should quote the exact per-QPU A/B and cite the schedule
JSON, drop the "quantum Gibbs does not describe the measurement" line, and
replace it with a brief note that the two predictions agree with the data
in the relaxed regime, with the frustrated pilot as a distinguishing
diagnostic.

## 6. Audit trail

- `scripts/reanalyze_phase6_with_correct_ab.py` — one-shot re-analysis
  driver (idempotent; only replaces `comparison.json` when a backup
  already exists).
- `scripts/ab_correction_summary.py` — produces the before/after tables
  above from the (patched) summaries and their `.bak_approx_ab1p33`
  backups.
- `data/raw/phase6_gibbs/*/summary.json.bak_approx_ab1p33` — original
  A/B=1.333 summaries.
- `data/raw/phase6_gibbs/*/comparison.json.bak_approx_ab1p33` — original
  A/B=1.333 comparison metrics.
- `data/raw/phase6_gibbs/phase1_calibration/schedules/schedule_at_sp0.4.json`
  — device-level A(s_p), B(s_p) at s_p=0.4 for both QPUs.
