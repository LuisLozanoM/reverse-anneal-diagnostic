# Phase 6 — Gibbs-fit Campaign

**Project**: locth1 — subsystem relaxation on a programmable quantum annealer.
**Purpose**: close the Nature gap identified by GPT Pro — upgrade the
"relaxation" claim to an "effective thermal marginal at the calibrated
β_eff" claim.
**Target venue**: Nature (flagship), with Nature Physics fallback.

## 0. Motivation

The original Phase 4 data in `data/raw/phase4/gibbs_fit/gibbs_fit_results.json`
attempted to fit an effective β via free-parameter MLE on the reduced
subsystem marginal. Every condition saturated at β = 9999.9997 (the optimizer's
upper bound) because the reduced distribution is peaked on the ferromagnetic
ground state and the log-likelihood is degenerate in that limit. The paper's
Discussion currently hides this as "future work". GPT Pro flagged this as
the crucial gap between Nature Physics and flagship Nature.

The Phase 6 fix:

1. **Fix β externally** via a single-qubit probe calibration at the pause
   point, not via free-parameter MLE.
2. **Compare measured P_S against theoretical predictions** at that fixed β
   — quantum reduced Gibbs of H(s_p), classical diagonal of H_P, conditional
   versions with fixed environment state, and an effective H_S^eff fit at
   the main-experiment scale.
3. **Check the fit fails exactly where relaxation fails** (Codex's success
   criterion #4).

## 1. Infrastructure

### 1.1 New module: `src/locth1/gibbs_comparison.py`

- `quantum_reduced_gibbs_marginal(h, J, S_indices, A_over_B, beta)` —
  diag of Tr_E[exp(-β H(s_p))/Z] in the z-basis, using real `eigh` on the
  full 2^N × 2^N Hamiltonian (N ≤ 12 is tractable, ~5 s per call on M3 Max).
- `classical_diagonal_marginal(h, J, S_indices, beta)` — classical Gibbs
  of H_P alone, enumerated over 2^N states (feasible up to N ≈ 20).
- `classical_conditional_marginal(h, J, S_indices, E_state, beta)` —
  conditional on a fixed environment state, enumerating only 2^|S| states;
  cheap at any |E|.
- `quantum_conditional_marginal(h, J, S_indices, E_state, A_over_B, beta)` —
  quantum version of the above, still needs full eigh so N ≤ 14.
- `fit_effective_H_S(P_S_measured, beta, edges=None, smoothing=0, l2_regularization=0)`
  — logsumexp-based NLL fit at fixed β via L-BFGS-B, materialises all 2^|S|
  states (no missing-bin issue), supports Laplace smoothing and L2 regularisation.
- `compute_gibbs_fit_metrics(...)` — D_TV, symmetric KL, magnetisation MAE,
  pair-correlation MAE.

**Codex review applied**: n inference includes S_indices; Hermiticity check
raises on genuine asymmetry; complex→real cast for 6× eigh speedup; fit is
L-BFGS-B on logsumexp NLL with shared eigendecomp between conditional and
unconditional paths.

### 1.2 New script: `scripts/phase6_gibbs_campaign.py`

Subcommands:

- `calibrate_beta_eff` — single-qubit probe calibration at s_p
- `calibrate_readout` — 6-qubit readout confusion-matrix calibration
- `run_phase2_exact` / `analyze_phase2_exact` — |S|=4 ED-regime sweep
- `run_phase3_bath` / `analyze_phase3_bath` — |S|=6 bath-ensemble (placeholder)

Each run supports `--dry-run` (classical sampling) to exercise the pipeline
without QPU time.

### 1.3 New config: `configs/phase6_gibbs.yaml`

Holds the calibration targets, the A(s)/B(s) schedule (approximate, pending
CSV from D-Wave), the QPU solver names, and the full 4D sweep parameters.

## 2. Calibration (Phase 1)

### 2.1 β_eff via single-qubit probe

Protocol:
- 3 probe qubits from `sampler.nodelist[:3]`
- h = 0.5, J = {}, `auto_scale=False`
- 5000 reverse-anneal reads per initial state (all-up, all-down)
- β_eff = ln(n_down / n_up) / (2 h)

Result (Advantage2_system1, 2026-04-10):

| Probe | n_up | n_down | β_eff |
|---|---|---|---|
| q0 | 8 | 9992 | 7.130 |
| q1 | 7 | 9993 | 7.264 |
| q2 | 7 | 9993 | 7.264 |

**β_eff = 7.219 ± 0.063** (mean ± std across 3 probes)

Previous paper value was 6.93 ± 0.20. The new calibration is ~4% higher,
consistent with calibration drift across the April 9 → April 10 rename.

### 2.2 A(s), B(s) schedule

`sampler.properties` does NOT include the anneal schedule (verified on
Advantage2_system1 — neither `anneal_schedule` nor `schedule` keys are
present). D-Wave publishes per-solver schedules as separate XLSX files
that must be manually downloaded. **Completed 2026-04-11**: the schedules
for Advantage2_system1 and Advantage_system6.4 have been downloaded and
parsed; the values at s_p = 0.4 are stored in
`data/raw/phase6_gibbs/phase1_calibration/schedules/schedule_at_sp0.4.json`.

| Solver | A(s_p=0.4) [GHz] | B(s_p=0.4) [GHz] | A/B |
|---|---|---|---|
| Advantage2_system1 | 0.9636 | 3.7024 | **0.2603** |
| Advantage_system6.4 | 0.5002 | 1.9320 | **0.2589** |

Phase 2 was originally run with approximate values
(A(0.4) ≈ 2.0 GHz, B(0.4) ≈ 1.5 GHz, A/B ≈ 1.333) pulled from
`scripts/phase5/run_ed.py`. The full Phase 6 quantum-Gibbs comparisons
have been re-computed with the device-calibrated A/B values above; see
`docs/phase6_ab_correction_findings.md` for the before/after tables and
`scripts/reanalyze_phase6_with_correct_ab.py` for the one-shot re-analysis
driver. The classical conditional Gibbs prediction is A/B-independent and
is unaffected by the correction; the quantum conditional Gibbs prediction
is **much tighter** at the corrected A/B and becomes the tighter target in
the frustrated (Edwards–Anderson bond disorder) pilot.

### 2.3 Qubit 4374 / Advantage2_system1.13 → Advantage2_system1 rename

**D-Wave announcement (2026-04-10)**: qubit 4374 on `Advantage2_system1.13`
was operating out of spec between 2026-04-08 21:30 PT and 2026-04-10 10:50
PT. The solver has been replaced by `Advantage2_system1` with qubit 4374
and its couplers removed from the working graph.

**Impact on Phase 6**: NONE. Our calibration and Phase 2 data were collected
on 2026-04-10 after 13:55 local (i.e., after the rename). All h5 files and
the calibration JSON explicitly store `solver = "Advantage2_system1"`. The
bad qubit is excluded from the working graph and cannot appear in any
embedding.

**Impact on the main paper's Phase 4 data (pre-existing)**:
- `scan_E_size`, `scan_crossover` collected 2026-04-08 ~08:00 PT — BEFORE
  the incident window. Safe.
- `disorder_averaging`, `pre_annealed_E`, `E_state_averaging`,
  `pause_time_kinetics` collected 2026-04-09 — POTENTIALLY within the
  incident window. Whether qubit 4374 was actually in the embedding is
  unclear without the embedding logs.

**Recommendation for Methods**: update the solver name from
`Advantage2_system1.13` to `Advantage2_system1`, and flag the April 9 data
as potentially affected by the incident if re-verification is not possible.

## 3. Phase 2 experimental design

### 3.1 Protocol

- **Subsystem**: |S| = 4 qubits
- **System sizes**: N ∈ {6, 8, 10, 12} (|E| ∈ {2, 4, 6, 8}); N=14 excluded
  because the 16384 × 16384 dense eigh is prohibitive
- **Coupling strengths**: λ ∈ {0.2, 0.3, 0.5} — spans the main paper's
  λ_c ≈ 0.2 crossover, with λ=0.3 as the "weaker-boundary" condition and
  λ=0.5 as the "strong-boundary" reference
- **Disorder strengths**: W ∈ {0.0, 0.5, 1.0}
- **Disorder seeds**: {42} at W=0, {42, 123, 456} at W > 0
- **S initial states**: all-up, all-down, random (per seed)
- **E preparation**: all-up (ordered low-energy bath)
- **Pause point**: s_p = 0.4
- **Pause time**: t_p = 100 μs
- **Reads per initial state**: 2000 (6000 pooled across initial states)
- **Graph geometry**: run TWICE, once with `random_regular` (logical 3-regular
  + minor-embedded) and once with `native` (Zephyr BFS subgraph)

### 3.2 Analysis pipeline

For each (λ, W, seed, N) condition:
1. Pool the three S initial-state marginals into `P_S_pooled`
2. Compute the memory order parameter M = max pairwise TVD across initial
   states; flag as **RELAXED** if M ≤ 0.05, **MEMORY** otherwise
3. Compute classical conditional Gibbs at β_eff = 7.219 with E fixed to all-up
4. Compute quantum conditional Gibbs at β_eff, A/B = **0.2603 (Advantage2)
   or 0.2589 (Advantage_system6.4)** — N ≤ 12 only. The original pass used
   the approximate A/B = 1.333 and has been superseded by the re-analysis in
   `scripts/reanalyze_phase6_with_correct_ab.py`.
5. Compute classical/quantum unconditional marginals for comparison
6. Record D_TV, symmetric KL, magnetisation MAE, pair-correlation MAE,
   and the pooled Shannon entropy (nats)

### 3.3 Total experiments

- random_regular: 3 λ × (1 + 3 + 3) seed-per-W × 4 N × 3 init = **252 QPU jobs**
- native Zephyr: same structure = **252 QPU jobs**
- **Total Phase 2: 504 QPU jobs, ~10 minutes wall time**

## 4. Phase 2 results

### 4.1 Relaxation counts

| Geometry | λ=0.2 | λ=0.3 | λ=0.5 | **total** |
|---|---|---|---|---|
| random_regular | 4/28 | 4/28 | 10/28 | **18/84** |
| native Zephyr | 5/28 | 3/28 | 2/28 | **10/84** |
| **combined** | 9/56 | 7/56 | 12/56 | **28/168** |

Key observation: native Zephyr has FEWER relaxed cases than random_regular
because the native BFS subgraph at small N is a dense ferromagnetic cluster
(Zephyr degree ~20) that resists thermalisation more than a 3-regular logical
graph with minor embedding. This is the opposite of what I expected.

### 4.2 Thermal-marginal agreement (the Nature claim)

**Every relaxed condition across both geometries** satisfies
D_TV(pooled_measured, classical_conditional_Gibbs@β_eff) below the
6000-read Poisson floor (~0.02):

| Geometry | n_relaxed | mean D_TV(c\|E) | max D_TV(c\|E) |
|---|---|---|---|
| random_regular | 18 | 0.0014 | 0.0173 |
| native Zephyr | 10 | 0.0010 | 0.0028 |
| **combined** | **28** | **0.0013** | **0.0173** |

**The agreement is perfect to shot-noise precision.** In every relaxed
condition, independent of graph topology, disorder, seed, size, and coupling
strength, the measured distribution matches the classical conditional Gibbs
at the independently calibrated β_eff.

### 4.3 Quantum comparison

> **Correction 2026-04-11.** The first pass of this section reported
> D_TV(quantum conditional) ≈ 0.3–0.8 using an approximate
> A(s_p)/B(s_p) ≈ 1.333. The actual per-QPU ratios from the solver
> schedule XLSX are 0.2603 (Advantage2) and 0.2589 (System 6.4), about
> 5× smaller. Every quantum-target comparison has been regenerated;
> see `docs/phase6_ab_correction_findings.md` for the full audit trail.
> The numbers below are the corrected values.

**Phase 2 native Zephyr (10 relaxed):** quantum conditional D_TV median
0.028, max 0.056; quantum unconditional median 0.031, max 0.52.

**Phase 2 random 3-regular (18 relaxed):** quantum conditional median
0.021, max 0.45; quantum unconditional median 0.068, max 1.00.

Classical conditional remains the tightest target in every relaxed
ferromagnetic case we tested, but the quantum conditional target is
now also consistent with the measurement, not an order-of-magnitude
negative result. The Phase 3A scale dependence of the quantum
conditional target is discussed in §6.1 and §6.4.

**Physical interpretation (updated):** the classical conditional Gibbs
target is tighter than the conditional quantum reduced Gibbs target in
every ferromagnetic regime because the return ramp from s_p back to s=1
adiabatically projects the quantum Gibbs state onto the classical Ising
eigenbasis at s=1, removing the residual transverse-field-induced
mixing that the conditional quantum diagonal would otherwise carry.
The externally calibrated β_eff absorbs the full protocol-level mapping
(reverse anneal → pause → return ramp → readout), so the relevant
theoretical target for the z-basis readout is the classical conditional
Gibbs marginal at that β_eff, not the pause-point quantum reduced Gibbs.
The conditional quantum reduced Gibbs is still useful as an
*independent* diagnostic that happens to agree with the measurement on
the clean Phase 3C ferromagnetic sweep (within 0.03 on both QPUs) and
**beats** the classical target on the frustrated Edwards–Anderson pilot
(§6.4).

### 4.4 Memory-retaining cases (negative control)

Across the full 494-condition Phase 6 main sweep, 381 conditions are
memory-retaining ($\mathcal{M}>0.05$). Their classical conditional
D_TV against the prepared-E target spans [0.003, 0.97] with median
0.35 and interquartile range [0.30, 0.62]. The fit fails in the
majority of memory-retaining cases, as expected, but 22/381 have
D_TV < 0.05 — these are "accidental pool matches" where the
three-initial-state average happens to resemble the classical target
even though the individual distributions differ significantly. The
memory order parameter and the thermal-marginal distance are therefore
genuinely independent diagnostics; reporting both separates ordinary
thermal relaxation, memory-retaining conditions, the small set of
relaxed-but-non-thermal trapping instances (§6.1), and these
accidental pool matches.

### 4.5 Limitation: ground-state-dominated regime

At β_eff = 7.2 with ferromagnetic couplings and the E-up boundary field, the
thermal Gibbs distribution is essentially a delta on the ferromagnetic ground
state. Of the 28 relaxed cases, only 2 have measurable entropy (H > 0.02 nats):

| λ | N | W | seed | H (nats) | D_TV(c\|E) |
|---|---|---|---|---|---|
| 0.3 | 6 | 1.0 | 123 | 0.021 | 0.0030 |
| 0.5 | 12 | 1.0 | 42 | 0.087 | 0.0173 |

These are the "real" tests of the Gibbs prediction: cases where the pooled
distribution has measurable spread, and the classical Gibbs prediction
still matches within shot noise. The other 26 relaxed cases are delta-like
matches on the ground state — consistent but not as informative.

**Implication for Phase 3**: to produce more non-trivial entropy cases, the
Hamiltonian should be scaled down (reducing effective β×E) or frustrated
(breaking ground-state degeneracy). This motivates Phase 3A below.

## 5. Phase 3 design

### 5.1 Phase 3A — scaled Hamiltonian at |S|=4

Same structure as Phase 2 but with `hamiltonian_scale = 0.5`: all
(h, J) values scaled by 0.5 before submission. Effective β×E ≈ 3.6 instead
of 7.2, producing pooled entropy 0.3–1.0 nats in the relaxed regime.

- 3 λ × 3 W × 7 seed combinations × 4 N × 3 init = 252 jobs × 2 geometries = **504 QPU jobs**
- Output: `data/raw/phase6_gibbs/phase3a_scaled_{randomreg,native}/`

Expected outcome: more non-trivial entropy cases, stronger Gibbs test.

### 5.2 Phase 3B — bath-ensemble fit at |S|=6

Main-experiment-scale test using the effective H_S^eff fit.

- |S| = 6, |E| = 50 (N_total = 56, matches main paper Phase 4)
- coupling_lambda = 0.5 (well above λ_c ≈ 0.2 from main paper)
- W ∈ {0.0, 0.5, 1.0}
- Seeds: {42} at W=0, {42, 123, 456} at W > 0
- 7 conditions × 3 S initial states = **21 QPU jobs**
- Graph: native Zephyr (matches main paper Fig 3 geometry)
- Analysis:
  - Classical conditional Gibbs at β_eff (cheap at any |E|)
  - Effective H_S^eff fit with 6 fields + 15 couplings = 21 parameters
  - Compare fitted (h̃, J̃) to expected boundary-shifted values
- No quantum conditional (ED infeasible)

Expected outcome: the thermal-marginal bridge holds at the main experimental
scale, and the fitted effective Hamiltonian is a small perturbation of the
expected boundary-shifted ferromagnet.

### 5.3 Phase 3C — disorder averaging at fixed N=12

Dedicated disorder-sweep to produce the "D_TV vs W" figure for the paper.

- N = 12 fixed, |S| = 4, λ = 0.5
- W ∈ {0.0, 0.25, 0.5, 0.75, 1.0, 1.25}
- Seeds: {42, 123, 456, 789, 1024} (5 seeds per W)
- 6 × 5 × 3 init = **90 QPU jobs**
- Graph: native Zephyr
- Analysis: per-seed D_TV, then seed-averaged with error bars

Expected outcome: the figure shows D_TV < 0.01 for W below arrest, rising
sharply for W above arrest — cleanly demonstrating "fit holds below arrest,
fails at arrest".

### 5.4 Total Phase 3 budget

- 3A: 504 jobs, ~10 min
- 3B: 21 jobs, ~1 min
- 3C: 90 jobs, ~3 min
- **Total: 615 jobs, ~15 min QPU wall time**

## 6. Phase 3 results

All three Phase 3 sweeps are done (2026-04-10 to 2026-04-11). Total new QPU
work: 252 (3A) + 42 (3B, both scales) + 180 (3C) = **474 additional QPU jobs**.

### 6.1 Phase 3A — scaled Hamiltonian at |S|=4, native

Phase 3A was eventually run at three Hamiltonian scales (0.35, 0.5,
0.75), each with the full 252-job structure (3 λ × 7 (W, seed) × 4 N
× 3 initial states = 84 physical conditions × 3 init = 252 reads).
Relaxation counts and classical conditional statistics:

| scale | relaxed | cl D_TV median | cl D_TV max | non-trivial entropy |
|---|---|---|---|---|
| 0.35 | 12/84 | 0.013 | 0.941 | several |
| 0.5  | 10/84 | 0.006 | 0.967 | 5 |
| 0.75 | 9/84  | 0.002 | 0.016 | a few |

The scaling produced more non-trivial entropy at smaller scales but
also exposed a **dynamical-trapping failure mode** that appears at
*two* Hamiltonian scales for the same physical instance:
(λ=0.2, N=8, W=1.0, seed 42, native Zephyr).

| scale | M | cl D_TV | q cond D_TV | basin |
|---|---|---|---|---|
| 0.35 | 0.0145 | 0.9409 | 0.5414 | all-down (m ≈ -0.97) |
| 0.5  | 0.0240 | 0.9674 | 0.3565 | all-down (m ≈ -0.97) |

Both fail the 0.05 relaxation threshold but have pooled distributions
that place ~98.5% weight on the high-energy all-down basin rather than
on the classical all-up ground state. A third relaxed condition at the
same (λ, N, W) but seed 123 and scale 0.35 is a borderline classical
miss (M=0.0085, cl D_TV=0.0513, q cond D_TV=0.2956) with the measured
magnetisation near +1 on every site — it is not a wrong-basin trap.

Direct enumeration of the conditional classical energies for the
scale-0.5 wrong-basin instance gives (in the scaled units):
- E(all-up)   = -1.7142 (the true classical ground state)
- E(all-down) = -1.1675 (higher by 0.5467)
- Classical Gibbs ratio P(up)/P(down) = exp(7.219 × 0.5467) ≈ 51.77

All three $S$ initial states (all-up, all-down, random) converge to
the *same* high-energy all-down basin, not the Gibbs-preferred all-up
basin. This is **not** an inference bug — verified by running
`classical_conditional_marginal` directly on the stored h/J — and is
**not** a readout-confusion artefact (the 64-state confusion matrix is
fidelity 1.0000, see §10). It is a genuine dynamical trapping: the
reverse-anneal + return-ramp dynamics lock onto a metastable
configuration that is not the thermal equilibrium.

**Scientific interpretation**: the memory order parameter
$\mathcal{M}<0.05$ is necessary but **not** sufficient for thermal
equilibration. The Gibbs comparison provides strictly more
information than $\mathcal{M}$ alone — it can distinguish thermal
relaxation from "all initial states converging to the same
non-thermal fixed point". The trapping is physical, not hardware
artefact: the same instance fails at two different Hamiltonian scales,
and the robustness panel (§phase6_robustness) confirmed that the
scale-0.5 instance remains trapped under 1 μs, 5 μs and 20 μs
return ramps and under four random spin-reversal gauges. For the
Nature paper, this is not a problem to hide; it's evidence that the
thermal-marginal bridge is a meaningful test that can fail in
specific, physically interpretable ways.

### 6.2 Phase 3B — bath-ensemble at |S|=6, |E|=50, native

21 QPU jobs at scale=1.0 + 21 jobs at scale=0.5 = 42 total.

**Unscaled**: 5/7 relaxed, max D_TV = 0.0005, zero non-trivial entropy cases.
**Scale=0.5**: 6/7 relaxed, max D_TV = 0.0003, zero non-trivial entropy cases.

At the main experimental scale (|S|=6, |E|=50 on native Zephyr), the relaxed
distribution is always a delta on the ferromagnetic ground state, even with
scaling. This is expected: at β_eff = 7.22 with 50 E qubits all pinning S to
+1 via -λ boundary couplings, the effective field on S is dominated by the
boundary and the ground state is nearly deterministic.

**Effective H_S^eff fit**: triggered by the entropy gate H > 0.05 nats
(pre-registered per Codex Q3). **Zero conditions** passed the gate because
every relaxed distribution has H < 0.01. The fit was not performed.

The Phase 3B result is therefore: **the classical conditional Gibbs prediction
at the externally calibrated β_eff agrees with the measurement to D_TV < 0.001
at the main experimental scale**, but the effective-Hamiltonian inference is
degenerate (delta distributions cannot distinguish different 21-parameter fits).
This is consistent with the Phase 2 finding extended to a larger subsystem.

### 6.3 Phase 3C — disorder-averaged sweep (the paper figure)

**180 QPU jobs** (6 W × 10 seeds × 3 initial states) on random_regular,
|S|=4, N=12, λ=0.5.

| W | relax rate | D_TV relaxed (mean ± std) |
|---|---|---|
| 0.00 | 10/10 (100%) | 0.0028 ± 0.0048 |
| 0.25 | 9/10 (90%) | 0.0006 ± 0.0012 |
| 0.50 | 7/10 (70%) | 0.0018 ± 0.0035 |
| 0.75 | 6/10 (60%) | 0.0018 ± 0.0039 |
| 1.00 | 7/10 (70%) | 0.0017 ± 0.0040 |
| 1.25 | 4/10 (40%) | 0.0036 ± 0.0048 |

**The two-panel story**:

Panel A (relaxation probability): drops monotonically from 100% at W=0 to
40% at W=1.25. This matches the main paper's disorder arrest (Fig 4a) — the
number of disorder realisations that successfully relax decreases with W.

Panel B (thermal marginal agreement in the relaxed subset): the D_TV stays
at **0.001–0.004 across all W values**. **When the subsystem relaxes, the
measured distribution is the classical conditional Gibbs at β_eff, regardless
of the disorder strength**. The Gibbs prediction is robust to disorder —
the only effect of disorder is to reduce the relaxation probability, not
change the form of the relaxed distribution.

**43 relaxed conditions out of 60**, mean D_TV = **0.0019**, max D_TV = **0.0163**.
**8 non-trivial entropy cases** (H > 0.02 nats). This is the strongest
dataset we have for the Nature claim.

## 7. The Nature claim (final, post-correction)

Across the **494-condition Phase 6 main sweep** on Advantage2
(Phase 2 native Zephyr, Phase 2 random 3-regular, Phase 3A at three
Hamiltonian scales, Phase 3B |S|=6 bath-ensemble at two scales, Phase
3C 10-seed disorder sweep), plus a **60-condition cross-QPU replication**
on Advantage_system6.4, the following result holds:

> **In the relaxed regime (memory order parameter $\mathcal{M}\le 0.05$),
> the measured subsystem distribution is quantitatively consistent with
> the classical conditional Gibbs marginal at the externally calibrated
> $\beta_\mathrm{eff}$, with $D_\mathrm{TV}$ below the multinomial
> shot-noise floor ($\approx 0.026$ for 6000 pooled reads and
> $K=2^{|S|}=16$ outcomes) in $110/113$ relaxed main-sweep conditions.**

The three exceptions are all at ($\lambda=0.2$, $N=8$, $W=1.0$) on
native Zephyr: the seed-42 instance traps at Hamiltonian scales 0.35
(cl $D_\mathrm{TV}=0.94$) and 0.5 (cl $D_\mathrm{TV}=0.97$) — the
wrong-basin pair discussed in §6.1 — and the seed-123 instance at
scale 0.35 is a borderline classical miss (cl $D_\mathrm{TV}=0.051$).

Supporting sub-claims (corrected numbers):

1. The agreement holds across $|S|\in\{4, 6\}$, $|E|\in\{2,4,6,8,50\}$,
   two graph geometries (random 3-regular and native Zephyr), three
   coupling strengths $\lambda\in\{0.2,0.3,0.5\}$, and four Hamiltonian
   energy scales ($\times0.35$, $\times0.5$, $\times0.75$, $\times1.0$).
2. **The thermal-marginal agreement is independent of disorder
   strength** in the 10-seed Phase 3C sweep: the relaxation probability
   drops from $10/10$ at $W=0$ to $4/10$ at $W=1.25$, but the mean
   classical $D_\mathrm{TV}$ in the relaxed subset stays at
   $0.6\!\times\!10^{-3}$ to $3.6\!\times\!10^{-3}$ across all $W$
   values (well below the shot-noise floor).
3. The fit is more permissive in the memory-retaining regime than a
   simple "fails where relaxation fails" picture would suggest: of 381
   memory-retaining main-sweep conditions, classical $D_\mathrm{TV}$
   spans $[0.003, 0.97]$ with median $0.35$, and $22/381$ satisfy
   $D_\mathrm{TV}<0.05$. These "accidental pool matches" are separate
   from the relaxed-but-non-thermal trapping failures and do not
   undermine the main result — they just mean the memory order
   parameter and the thermal-marginal distance are genuinely
   independent diagnostics.
4. The **classical conditional Gibbs** target is tighter than the
   **conditional quantum reduced Gibbs** target in every relaxed
   ferromagnetic phase, but with the **correct** per-QPU
   $A(s_p)/B(s_p)=0.2603$ (Advantage2) / $0.2589$ (System 6.4) from
   the solver schedule XLSX, the quantum conditional target is also
   consistent with the measurement on the Phase 3C ferromagnetic sweeps
   (median $D_\mathrm{TV}\approx0.015$, max $\approx0.028$ on both
   QPUs). The quantum conditional target is *looser* on the Phase 3A
   scaled sweeps — median $0.22$ at scale $0.35$, $0.11$ at scale
   $0.5$, $0.047$ at scale $0.75$ — because scaling $H_P$ by $s$
   leaves $A(s_p)$ unchanged and multiplies $B(s_p)H_P$ by $s$, so
   the effective transverse-to-longitudinal ratio $A/(sB)$ seen by
   the reduced Gibbs state grows with $1/s$.
5. **Frustrated (Edwards–Anderson) pilot flips the ordering**: on the
   two relaxed conditions in the $\pm J$ bond-disorder pilot (15
   conditions, $\lambda=0.5$, $N=12$, $|S|=4$), the classical
   conditional target has $D_\mathrm{TV}=0.32$ on average (max $0.64$)
   while the conditional quantum reduced Gibbs target has
   $D_\mathrm{TV}=0.029$ on average (max $0.045$). The quantum target
   beats the classical target by an order of magnitude in the
   multi-modal regime where the classical ground state is degenerate
   and tunneling between basins matters. The sample is small; a
   larger frustrated sweep is obvious follow-up.
6. The **dynamical-trapping caveat is a pair (not a singleton)**.
   Two relaxed Phase 3A conditions, both at
   ($\lambda=0.2$, $N=8$, $W=1.0$, seed 42, native Zephyr), show
   $\mathcal{M}\le 0.024$ but classical $D_\mathrm{TV}\ge 0.94$; the
   two correspond to Hamiltonian scales 0.35 and 0.5. Both converge to
   the higher-energy all-down basin. The trapping is robust under
   return-ramp and gauge averaging (§phase6_robustness). This
   demonstrates that $\mathcal{M}\to0$ is necessary but not sufficient
   for thermal equilibration; the Gibbs comparison adds strictly more
   information.
7. **Unconditional** quantum reduced Gibbs remains a poor target for
   the ordered-environment experiments (median $D_\mathrm{TV}\approx0.5$
   on Phase 3C Advantage2), as expected — averaging over the global
   $Z_2$ sectors cannot encode the prepared all-up boundary.

## 8. Phase 6 total QPU budget (as run)

| Phase | QPU jobs | Reads per job | Total reads |
|---|---|---|---|
| Calibration (β_eff probes) | 6 | 5000 | 30,000 |
| Phase 2 random_regular | 252 | 2000 | 504,000 |
| Phase 2 native | 252 | 2000 | 504,000 |
| Phase 3A scaled native | 252 | 2000 | 504,000 |
| Phase 3B |S|=6 unscaled | 21 | 2000 | 42,000 |
| Phase 3B |S|=6 scale=0.5 | 21 | 2000 | 42,000 |
| Phase 3C disorder sweep | 180 | 2000 | 360,000 |
| **Total** | **984** | | **~1.99M reads** |

Wall-clock QPU time: approximately 30–40 minutes across 2026-04-10 to 2026-04-11.

## 9. Phase 3A energy-scale extension (2026-04-11)

Following Codex's recommendation, we extended Phase 3A to
`scale ∈ {0.35, 0.5, 0.75}` (the `1.0` case is Phase 2 on native Zephyr).
504 additional QPU jobs (252 per scale).

| scale | relaxed / total | mean D_TV(c\|E) | max D_TV | non-trivial H |
|---|---|---|---|---|
| 0.35 | 12 / 84 | 0.098 | 0.941 | 9 |
| 0.50 | 10 / 84 | 0.103 | 0.967 | 5 |
| 0.75 | 9 / 84  | **0.005** | **0.016** | 4 |
| 1.00 (Phase 2 native) | 10 / 84 | 0.001 | 0.003 | 0 |

**Key observations**:

1. **Smaller scales produce more non-trivial entropy cases** — 9 at scale=0.35
   vs 0 at scale=1.0. This confirms the energy-scale scan is a valid tool for
   moving out of the ground-state-dominated regime.

2. **Smaller scales also introduce dynamical-trapping anomalies** — both
   scale=0.35 and scale=0.5 have one case each with D_TV > 0.9 in the
   relaxed subset. At scale=0.75 there are no anomalies and the mean
   D_TV stays at 0.005 (still below shot-noise floor).

3. **The thermal-marginal agreement is robust to the energy scale** in
   the well-behaved cases — at scale=0.75, 9 relaxed conditions with mean
   D_TV = 0.005 and 4 non-trivial entropy cases with max D_TV = 0.016.

The systematic scale scan therefore strengthens the main Phase 6 claim
while also revealing the scale-dependent dynamical-trapping failure mode
as a characterisable edge case, not a systematic bias.

## 10. Readout confusion calibration (2026-04-11)

Per Codex's ranked-top recommendation for robustness checks.

**Protocol**: For each of $2^{|S|}=64$ subsystem basis states, prepare the
system in that state as the `initial_state` argument and run the same
reverse-anneal schedule as the main experiment with $s_p = 0.95$ (very
shallow pause) and $t_h = 1.0\,\mu$s (minimal hold to avoid transverse-field
mixing). 200 reads per target state, 12{,}800 reads total, one QPU job.

**Result**: **fidelity = 1.0000 on every one of the 64 target basis states.**

At 200 reads per target, this upper-bounds the per-target readout error at
$<1/200 = 0.5\%$.  The readout error contribution to any $D_\mathrm{TV}$
measurement with $\ge 6000$ pooled reads is therefore much smaller than
the Poisson noise floor $\sim 0.01$-$0.02$, and applying the (identity)
confusion-matrix correction changes the Phase 2/3 $D_\mathrm{TV}$ values
by less than the reported uncertainty.

**Initial bug caught during this run**: my first version of the readout
calibration used `t_hold = 0` which produces duplicate schedule times and
is rejected by the D-Wave API (`SolverFailureError: invalid anneal_schedule`).
My second version used forward annealing, which the user correctly rejected
as inconsistent with the main paper's reverse-anneal protocol. The final
version uses reverse annealing with `s_p = 0.95`, `t_h = 1.0`\,\mu$s. Also
fixed: the first result analysis gave a nonsensical $1/8$ mean fidelity
because `samples[:, 0]` was being interpreted as logical qubit 0, whereas
`EmbeddingComposite` returns columns in the embedding-specific order given
by `sampleset.variables`.  The fix uses `compute_marginal` with the
explicit `variables=` argument.

## 11. Codex interpretation and manuscript review (2026-04-11)

Codex reviewed both the physical interpretation ("what happened to
quantum Gibbs") and the manuscript draft for the new §Effective thermal
marginal subsection, flagging 15 real issues in the first draft including
an internal contradiction between "every relaxed condition passes" and
the dynamical-trapping outlier.  Codex's red-team-revised draft is saved
at `manuscript/phase6_results.tex` (157 lines) and is ready to integrate
into `main.tex` between §Disorder arrests relaxation and §Classical and
small-system controls.

Key Codex recommendations that were applied:

- Replaced "thermal/equilibration" language with "conditional Boltzmann
  marginal" and "consistent with".
- Clarified that $\beta_\mathrm{eff}=7.22\pm0.06$ is the Phase 6
  recalibration (post-rename) and does NOT supersede the main-paper
  $6.93\pm0.20$ value.
- Fixed the internal contradiction: "all but one" or "excluding the
  trapping outlier".
- Softened the quantum reduced Gibbs paragraph: it is a diagnostic
  negative comparison, not proof of what the pause-point state "is".
- Qualified disorder range ("within the thermal-marginal sweep"), $Z_2$
  symmetry claims, shot-noise estimator references, and the
  metastability language.

Codex also ranked the remaining experimental gaps by importance:

1. **Readout confusion calibration** — DONE (this section).
2. **Return-ramp robustness** — DONE (1/5/20 μs return ramps on 3
   representative cases, `phase6_robustness`).
3. **Energy-scale scan** — DONE (scale ∈ {0.35, 0.5, 0.75, 1.0}).
4. **Gauge averaging** — DONE (4 random gauges per test case in
   `phase6_robustness`).
5. **Cross-QPU replication on Advantage_system6.4** — DONE
   (`phase3c_disorder_sweep_advantage_system64`, 60 conditions; 22/60
   relaxed).
6. **Frustrated Hamiltonian pilot** — DONE (`phase6_frustrated`, 15
   conditions).
7. **Exact A(s)/B(s) from solver schedule XLSX** — DONE
   (`data/raw/phase6_gibbs/phase1_calibration/schedules/schedule_at_sp0.4.json`),
   quantum-Gibbs metrics re-computed at the corrected A/B.

## 12. Open questions

1. **A(s_p), B(s_p) from the schedule** — **resolved 2026-04-11**. The
   corrected per-QPU A/B is 0.260 (Advantage2) and 0.259 (System 6.4),
   roughly 5× smaller than the approximate 1.333 that was used in the
   original quantum-Gibbs pass. Re-running the quantum comparisons with
   the correct A/B lowers the median quantum conditional D_TV in the
   Phase 3C ferromagnetic sweep from ≈ 0.5 to ≈ 0.015 (Advantage2) and
   from ≈ 0.6 to ≈ 0.015 (System 6.4). Classical conditional Gibbs
   remains the tighter target in the ferromagnetic regime, but in the
   frustrated pilot the ordering is reversed: classical conditional D_TV
   is 0.32 vs quantum conditional D_TV = 0.029 on the two relaxed
   conditions. See `docs/phase6_ab_correction_findings.md` for the full
   before/after.

2. **Phase 3B effective H_S^eff fit**: when the relaxed |S|=6 distribution
   is delta-like on the ground state (expected at β=7.2), the fit is
   degenerate. Does the L2 regularisation / Laplace smoothing handle this
   gracefully, or should we only fit cases with entropy > 0.02?

3. **Phase 3A scaling factor**: 0.5 is a guess. Sweep 0.25, 0.5, 0.75 to
   see which gives the best "non-trivial but relaxed" regime?

4. **Is a frustrated Hamiltonian a better Phase 3A?** — mixed-sign couplings
   J ∈ {-1, +1} with 50/50 probability would create genuine
   frustration and natural multi-minima distributions, which is a stronger
   test than pure scaling. But it's a different physics claim.

5. **Readout confusion-matrix calibration** is implemented but not yet run.
   For 6000-read ensembles at 16–64 states, the Poisson noise dominates
   over the ~10^-3 per-qubit readout error, so we haven't needed it.
   Should we run it anyway as a belt-and-suspenders for the paper?

## 7. Files

```
configs/phase6_gibbs.yaml              # campaign parameters
src/locth1/gibbs_comparison.py         # comparison / fit module
tests/test_gibbs_comparison.py         # 14 unit tests
scripts/phase6_gibbs_campaign.py       # driver with 5+ subcommands

data/raw/phase6_gibbs/
├── phase1_calibration/
│   ├── advantage2_properties.json     # full sampler.properties dump
│   └── beta_eff.json                  # beta_eff = 7.219 ± 0.063
├── phase2_exact/                      # native Zephyr sweep (current)
│   ├── summary.json
│   ├── comparison.json
│   └── lam{λ}_W{W}_seed{s}_N{N}_{init}.h5  × 252
├── phase2_exact_randomreg/            # random 3-regular sweep
│   ├── summary.json
│   ├── comparison.json
│   └── lam{λ}_W{W}_seed{s}_N{N}_{init}.h5  × 252
└── phase3_{a,b,c}_...                 # pending
```
