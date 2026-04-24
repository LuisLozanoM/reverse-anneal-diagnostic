# Kimi Code CLI Assessment — locth1 Nature Submission

**Date:** 2026-04-20
**Scope:** Full codebase review, paper review (main.tex + supplementary.tex), experimental design audit, and cross-check against the quantum-annealer-playbook skill.
**Verdict:** Nature-level claim with Nature-level rigor in design, but with one critical operational oversight and several gaps that must be addressed before submission.

---

## 1. Executive Summary

The core scientific logic is sound: partition the annealer into S and E, show initial-state independence grows with |E| and λ, arrest it with disorder, and demonstrate that the relaxed readout matches a calibrated conditional Gibbs law. The codebase is modular, extensively tested (42/42 passing), and the supplementary materials are exceptionally thorough. The frustrated pilot and dynamical-trapping analysis show real scientific maturity.

However, the main QPU experiments appear to have run with D-Wave's default `auto_scale=True`, while your β_eff calibration and classical baselines assume the Hamiltonian was programmed as-written. According to your own quantum-annealer-playbook skill, this is a non-negotiable violation that silently invalidates quantitative classical-vs-QPU comparisons. You need to either verify R ≈ 1 for every condition or re-run with `auto_scale=False`.

---

## 2. What You Are Doing Correctly (Major Strengths)

| Area | Assessment |
|------|------------|
| **Experimental design** | The S/E partition with tunable λ, |E|, W, and s_p is exactly the right upgrade from a global "memory loss" experiment to a true subsystem-thermalization study. |
| **Code quality** | Clean modular architecture (hamiltonians, observables, analysis, gibbs_comparison, classical baselines). 42 unit tests, all passing. Good type hints and docstrings. |
| **Statistical methodology** | Bootstrap resampling for error bars, TVD as the distance metric, memory order parameter M as a scalar diagnostic — all well-defined and consistently applied. |
| **Classical baselines** | You built four baselines (ED, Lindblad, Glauber, SVMC) and ran them systematically. This is far above the typical standard. |
| **Thermal-marginal analysis** | The conditional Gibbs target P_S^th(σ_S | σ_E = 1) is the physically correct object for fixed-E preparations, and you compare it against both classical and quantum reduced Gibbs targets. |
| **Failure-mode analysis** | Identifying "wrong-basin trapping" (seed-42, λ=0.2, N=8) as a distinct dynamical failure mode, separate from simple non-relaxation, is sophisticated and reviewer-friendly. |
| **Cross-QPU replication** | Running on both Advantage2 and System 6.4 with different energy scales strengthens the claim that the effect is Hamiltonian-driven, not device-specific. |
| **Supplementary depth** | The supplementary.tex is outstanding: schedule coefficients, per-sub-experiment D_TV tables, robustness panels, return-ramp controls, and gauge checks. |

---

## 3. Critical Issues (Fix Before Submission)

### 🔴 Issue 1: auto_scale — The Silent Hamiltonian Mismatch

**The problem:** Your `reverse_anneal.py` driver does **not** set `auto_scale=False`. D-Wave therefore defaults to `auto_scale=True`, which silently rescales (h, J) to fit the hardware's programmable ranges. You only enforce `auto_scale=False` in the Phase 6 β_eff calibration probes (`scripts/phase6_gibbs_campaign.py`, lines 741 and 2519).

Your own **quantum-annealer-playbook** skill states this is non-negotiable:

> "`auto_scale=True` silently rescales `h`/`J` to fit the solver's programmable range. This makes every quantitative comparison (classical vs QPU, one prefactor vs another, one seed vs another) meaningless."

**Why this matters for your paper:**
- Your in-situ β_eff is measured with `auto_scale=False` (R=1).
- Your main experiments (Phase 3/4, Figs. 2–3) likely ran with `auto_scale=True`.
- If R ≠ 1, the QPU programmed R·h and R·J, while your classical baselines (Glauber, ED, SVMC) simulated the nominal (h, J).
- This means the classical "kill" comparisons are **not at the same Hamiltonian** as the QPU.
- The conditional Gibbs predictions used for the thermal-marginal test would also be at the wrong effective temperature if R is not accounted for.

**What you must do:**
1. **Check the raw QPU sampleset metadata** for every main-text condition. D-Wave's `sampleset.info` usually contains the actual programmed coefficients under a `problem_data` or `timing` field. Extract the rescaling factor R (or confirm it was 1.0).
2. **If R ≠ 1 for any condition**, you have two options:
   - **Preferred:** Re-run the main Phase 3/4 sweeps with `auto_scale=False` (and verify (h, J) fit within `h_range`/`j_range` — with W ≤ 2.0 and |J| ≤ 1.0 they likely do).
   - **Fallback:** Extract R per condition, rescale the classical targets by R, and explicitly state in the Methods that all reported β_eff values are native-scale values multiplied by the condition-specific R.
3. **Add `auto_scale=False` as a hard-coded default** in `reverse_anneal.py` and add a validation check that raises if it is ever `True`.

---

### 🔴 Issue 2: β_eff Calibration — No Cross-Check

You use a single-qubit probe with two estimators (all-up and all-down population ratios averaged over 3 qubits). The playbook recommends **at least two independent estimators** (e.g., single-qubit probe + maximum pseudolikelihood on a small test problem) that agree within ~10%.

**Action:** Add an MPL or fast-effective-temperature cross-check on a small random 3-regular problem (N=8–12) and report the agreement in the Methods. If they disagree, diagnose before proceeding.

---

### 🔴 Issue 3: Gauge Averaging in Main Experiments

Your main experiments use a **fixed gauge** (the natural embedding mapping). The supplementary includes a 4-gauge robustness panel on 3 test cases, but the headline data in Figs. 2–4 are single-gauge.

D-Wave has well-documented per-qubit ICE (Intrinsic Control Error) biases that spin-reversal gauges are designed to cancel. A referee will ask whether the observed crossovers are robust to gauge averaging.

**Action:** Either:
- Run a gauge-averaged subset of the main sweeps (e.g., the |E| sweep and the (λ, W) grid at λ=0.5) and show that M and D_TV are gauge-independent within bootstrap error, or
- Explicitly justify the fixed-gauge choice in the Methods (e.g., "disorder averaging over seeds with both positive and negative dominant fields acts as a partial gauge average") and point to the supplementary robustness panel.

---

### 🟡 Issue 4: Underpowered Frustrated Pilot

The Edwards–Anderson frustrated pilot has only **2 relaxed instances out of 15**. The quantum conditional target outperforms the classical by an order of magnitude on these two, but a sample of 2 is not statistically persuasive.

**Action:**
- Frame this more cautiously in the main text. Current wording ("an order-of-magnitude-tighter agreement… but this subset is too small to establish the effect") is actually good, but make sure the **abstract and discussion** do not overclaim quantum effects in the frustrated regime.
- Consider expanding the frustrated campaign to N=12, |S|=4 with 20–30 seeds if QPU time allows, or move the frustrated result to a "future work" paragraph in the Discussion.

---

### 🟡 Issue 5: Scaling Collapse Failure

Supplementary Sec. 7.8 shows that rescaling Advantage2 couplings by α = β_eff^S6.4 / β_eff^A2 = 0.594 does **not** collapse the cross-QPU λ_c data onto a single curve; a residual 3× gap remains.

This directly weakens the **universality** claim (Criterion 5 for Nature in your own `analysis.md`).

**Action:**
- Be more explicit in the main text that a full quantitative collapse requires matching **both** β_eff **and** the transverse-field ratio A(s_p)/B(s_p), which differ between QPUs at the same nominal s_p.
- Do not claim "universality" in the strict scaling-collapse sense; instead frame the cross-QPU agreement as "consistent qualitative behavior with thresholds shifted by the measured energy-scale ratio."

---

## 4. Medium-Priority Gaps

| Issue | Why it matters | Suggested fix |
|-------|---------------|---------------|
| **ED time-unit mapping** | The main text says "Time is converted to natural units using the energy scales at the pause point" but never gives the conversion factor. A referee will ask how t_p = 100 μs maps to the dimensionless t in `expm_multiply(-1j * H * t)`. | Add one sentence to Methods: "The dimensionless ED time is t_ED = t_p · B(s_p) / (2ℏ)", where t_p is in seconds and B(s_p) is in GHz. (Or whatever your actual conversion is — verify it matches the code's `GHz_US_TO_NATURAL` constant.) |
| **Chain-break analysis** | Embedding is a major difference between your theoretical models and the QPU. You state chain-break rates are "a few percent" but do not correlate them with relaxation quality. | Add a supplementary figure or table: chain-break rate vs. M and D_TV for the embedded conditions. If chain breaks correlate with worse thermal-marginal fits, that's important to report. |
| **Lindblad absent from main text** | You built a Lindblad solver (nice work) but it only appears in the supplementary. If it was only tractable for N ≤ 12 and shows the same qualitative trends as ED, that's fine, but the main text should at least mention it. | Add one sentence in the "Classical and small-system controls" subsection: "A Lindblad master-equation treatment with local thermal jump operators (Supplementary Sec. 3.2) confirms the same qualitative trends for N ≤ 12." |
| **Low relaxation fraction** | Only 113/494 Phase 6 conditions (23%) satisfy M ≤ 0.05. This is not a bug — you deliberately swept a wide parameter space — but a referee may ask why most conditions *don't* relax. | Add a sentence in the Discussion acknowledging this: "The majority of conditions in the broad Phase 6 sweep retain memory, reflecting the finite-size, finite-time, and finite-coupling nature of the experiment; the thermal-marginal law holds specifically within the relaxed subset identified by M." |
| **Native-graph disorder threshold** | Supplementary Table 7 shows native Zephyr subgraphs relax at *all* tested W up to 2.0 because of much higher boundary connectivity (~83 edges vs. ~8 for random 3-regular). This suggests the disorder arrest threshold is **geometry-dependent**. | Discuss this explicitly: "The lower connectivity of the random 3-regular logical graph makes it a more stringent test of disorder arrest; the native-graph results constrain embedding-induced systematics but do not contradict the main conclusion." |

---

## 5. Code-Level Observations

### Correct and well-implemented
- **`gibbs_comparison.py`**: The quantum reduced Gibbs marginal uses the correct Hamiltonian convention (H = -A_overB Σ σ_x + H_P) with β_eff applied consistently. The partial-trace bit ordering matches `exact_diag.py`.
- **`exact_diag.py`**: The sparse Kronecker-product construction and `expm_multiply` time evolution are standard and correct. The B_s = 2.0 convention in `gibbs_comparison.py` properly cancels the B(s_p)/2 prefactor.
- **`observables.py`**: Gibbs fit uses MLE with bounded optimization; chi-squared is computed correctly. Bootstrap is used for error bars.
- **`classical/glauber.py`**: Detailed balance is correctly implemented; the test suite validates it against the 1D Ising equilibrium.

### Missing / needs attention
- **Lindblad tests**: There are no unit tests for `lindblad.py`. Add at least: (a) single-qubit thermalization reproduces expected T_1 ratio, and (b) two-qubit ferromagnet relaxes to correct thermal bias.
- **Metadata extraction**: `reverse_anneal.py` extracts `timing` and basic solver metadata but does **not** save `sampleset.info['problem_data']`, which would contain the actual programmed (h, J) when `auto_scale=True`. This is the forensic data you need to retroactively compute R.
- **Feature-based solver selection**: The code hardcodes solver names (`Advantage2_system1.13`) in configs. You already experienced the rename to `Advantage2_system1`. Switch to feature-based selection (`topology__type`, `name__prefix`) as recommended in your playbook.

---

## 6. Strategic Recommendations

### Immediate actions (this week)
1. **Audit `auto_scale`**: Check every main-text QPU run. If `auto_scale=True`, extract R from `sampleset.info` or re-run the critical sweeps with `auto_scale=False`.
2. **Lock `auto_scale=False`** in `reverse_anneal.py` as a hard default.
3. **Add a β_eff cross-check**: Run MPL on a small N=8 problem and compare to your single-qubit probe value.

### Before submission
4. **Gauge-averaged main data**: Run at least the |E| sweep and W sweep with 4 gauges and show equivalence to fixed-gauge results.
5. **Expand chain-break analysis**: Correlate chain-break fraction with M and D_TV for embedded conditions.
6. **Tone down frustrated claims**: Ensure the abstract and discussion do not imply the frustrated regime is established; keep it as a "motivating pilot."
7. **Clarify universality language**: Remove or soften any claim of "quantitative collapse" between QPUs; emphasize qualitative consistency instead.

### If you have extra QPU time
8. **Larger frustrated campaign**: 20–30 seeds at p_S = 0.5, W=1.0 to establish the quantum-vs-classical separation with statistical power.
9. **Native-graph cross-QPU**: Run the native Zephyr |S|=6, |E|=50 sweep on System 6.4 to strengthen the embedding-independence claim.

---

## 7. Final Word

The conceptual and experimental architecture of this paper is **genuinely Nature-worthy**. The subsystem-bath design, the conditional thermal-marginal test, and the systematic classical-kill campaign are all at the level required for a top journal.

The single biggest risk is the **`auto_scale` question**. If the main experiments ran with `auto_scale=True` and R ≠ 1, the classical comparisons are technically invalid, and a sharp referee (or editor) could desk-reject on that basis. Fix that, shore up the gauge averaging, and you have a very strong submission.
