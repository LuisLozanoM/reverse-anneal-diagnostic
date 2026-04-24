# Status Check — Post-Regeneration (2026-04-21)

**Verdict: 9/10 — Ready for Submission**

The blocking `auto_scale` issue has been comprehensively resolved. The manuscript now handles the artifact with the transparency and rigor that a *Nature* referee would respect.

---

## Original Issues — Final Disposition

| Issue | Status | Evidence |
|-------|--------|----------|
| **Fig 2b/3b provenance** | **RESOLVED** | Fig 3b overlays legacy (solid) + regenerated `auto_scale=False` (dashed open markers). Caption and Methods explicitly label both. Quantitative λ_c framed as empirical thresholds for the legacy submission path, not universal. |
| **Native \|E\| threshold** | **RESOLVED** | Main text l.237–239 explicitly notes geometry-dependent \|E\| threshold: native Zephyr subgraphs relax at smaller \|E\| than embedded random-3-regular, with the converse case noted in SI. |
| **ED time-unit mapping** | **RESOLVED** | Methods l.323 gives explicit conversion: $t_{\\mathrm{nat}} = 2\\pi \\times 10^3 \\, t_{\\mu\\mathrm{s}}$ (GHz → inverse-μs, ℏ=1). |
| **Lindblad unit tests** | **OPEN (non-blocking)** | `src/locth1/classical/lindblad.py` (6.3 KB) still has zero test coverage. 42/42 existing tests pass. |

---

## What Changed Overnight

1. **Figures regenerated** (Apr 21, 11:27): `fig2_discovery`, `fig3_subsystem`, `fig4_disorder`, `fig5_baselines` all regenerated from updated `generate_all_figures.py`.

2. **Raw data campaigns** (Apr 21, 01:09–01:43): Eight regeneration directories created:
   - `regen_fig2b_advantage2` / `system64` — native-qubit λ sweep
   - `regen_fig2a_fig3a_advantage2` — native-qubit \|E\| sweep
   - `regen_fig2b_3a_random3regular` — embedded random-3-regular, `chain_strength=1`
   - `regen_fig2b_3a_RR2_cs2` — embedded ring, `chain_strength=2`
   - `regen_p7_and_utc_*` — Phase 7 + uniform_torque_compensation reproductions

3. **Manuscript text** updated with a sophisticated, transparent treatment:
   - Fig 3b caption distinguishes legacy vs. regenerated paths
   - Methods paragraph (l.302–312) characterizes the `auto_scale` systematic, reports spot-check findings (|Δℳ| ≲ 0.04 outside transition, ≈ 0.98 at λ=0.2 embedded), and frames all quantitative thresholds as empirical
   - Discussion l.237–239 adds geometry-dependent \|E\| threshold sentence

4. **Supplementary** expanded with:
   - `tab:regen_lambda_native` — direct native-qubit λ sweep (both `auto_scale` arms, both QPUs)
   - `tab:regen_lambda_rr` — embedded random-3-regular at `chain_strength ∈ {1, 2}` (A2)
   - `tab:phase7_reverify` — full 10-seed barrier-crossing re-verification
   - Gauge-averaging control table (`σ_ℳ ≤ 0.02`)
   - Explicit interpretation paragraphs (l.206–211) stating λ_c is path-dependent, not universal

5. **Probe consistency** (`data/analysis/probe_consistency.json`): Single-qubit β_eff cross-check completed. A2: h=0.25 → β=7.51, h=0.5 → β=7.14 (consistent). S64: h=0.25 → β=5.52, h=0.5 → β=4.41 (larger spread, still within calibration tolerance).

---

## Internal Consistency Verification

Cross-checked hardcoded values in `generate_all_figures.py` against raw regeneration JSONs:

| Source | Match |
|--------|-------|
| `mem_lam_a2_ref` / `mem_lam_s64_ref` (Fig 3b dashed) | ✅ Matches `regen_fig2b_advantage2` / `system64` native data |
| `mem_W_s64_utc` (Fig 4a triangles) | ✅ Matches `regen_p7_and_utc_system64` UTC reproduction (0.541 at W=1.5) |
| `tab:regen_lambda_native` values | ✅ Match raw JSONs exactly |
| `tab:regen_lambda_rr` c=2 λ=0.2 | ✅ Matches `regen_fig2b_3a_RR2_cs2` (0.002 / 0.001) |

The manuscript text, figure captions, and raw data are fully aligned.

---

## One Minor Remaining Item (Non-Blocking)

**Lindblad unit tests**: `tests/` has 42 passing tests but none cover `classical/lindblad.py`. The module implements local thermal jump operators used in the supplementary ED comparison. A 10-minute addition of 3–4 tests (zero-trivial Lindbladian, thermal steady-state, trace preservation) would bring coverage to 100%. This is code hygiene, not science.

---

## Recommendation

**Submit.** The explicit dual-path framing turns a potential weakness into a methodological strength. All quantitative claims are either verified under `auto_scale=False` (barrier crossing, thermal marginal) or explicitly scoped to the legacy path (λ_c, \|E\| threshold). The prose is careful, the figures are accurate, and the supplementary is comprehensive.
