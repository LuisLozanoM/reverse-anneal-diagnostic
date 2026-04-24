# Final Pre-Submission Check — Response to Post-Addendum Changes

**Date:** 2026-04-21
**Context:** User implemented changes since the addendum and asked for a final go/no-go assessment before Nature submission.
**Changes described by user:**
1. Phase 7 QPU re-verified at `auto_scale=False` on both QPUs (10 seeds each, `chain_strength=2`). 8/10 relax on Advantage2, 7/10 on System 6.4 — consistent with or exceeding legacy 70%/50% rates. See Supp. Tab. `tab:phase7_reverify`.
2. CPU Glauber re-verified on exact same nominal $(h, J)$ as QPU submissions — 0/10 at device $T$, 100% at $T=0.5$. Temperature gap ~3.2× on Advantage2, ~1.8× on System 6.4. Same qualitative finding as legacy 4.8×/1.7×. Documented in Supp.
3. Disorder-arrest reproduced at `uniform_torque_compensation` chain strength on System 6.4: $\mathcal{M}=0.54$ at $W=1.5$ (legacy reported 0.50 at same working point).
4. Removed MPL cross-check from Supp per explicit recommendation — MPL is the wrong diagnostic for a conditional-state experiment.
5. Methods paragraph rewritten to explicitly cover Phase 7 `auto_scale` scope (not just Phase 3/4).
6. Code cleanup: `eval(k_str)` replaced with `ast.literal_eval` in 8 call sites; `hash()` seeding replaced with deterministic offsets; `reverse_anneal.py` now records `embedding_context` and `auto_scale` in metadata; reverse-anneal driver defaults to `auto_scale=False`.

---

## (a) Do these changes address the addendum's critical issues?

| Addendum Item | Status | Notes |
|---------------|--------|-------|
| **Lock `auto_scale=False` as hard default** | ✅ Fully addressed | Code + metadata + driver defaults all fixed. |
| **Add Methods paragraph on `auto_scale`** | ✅ Fully addressed | Explicit scope covering Phase 7 and legacy Phase 3/4 is the right level of transparency. |
| **Re-run Fig. 2b (lambda sweep) & Fig. 3b (crossover diagram)** | ⚠️ **Not confirmed by user** | User described Phase 7 re-verification, disorder-arrest spot check, and Glauber re-run — but did **not** state that the **main-text lambda sweep** (Fig. 2b) or the **2D $(\lambda, W)$ crossover grid** (Fig. 3b) were regenerated with `auto_scale=False`. Embedded spot checks show the crossover location shifts by $\Delta\lambda \approx 0.05$–$0.1$. If those two figures are still the legacy `auto_scale=True` data, they are the last artifact remaining. |
| **Spot-check $|E|$ and $W$ sweeps at $\lambda=0.5$** | ✅ Addressed | Disorder-arrest reproduced on System 6.4; $\lambda=0.5$ spot checks showed agreement earlier. |
| **Single-qubit probe consistency check** | ❌ Not mentioned | Still recommended: run probe at $h = 0.25$ and $h = 0.5$; verify same $\beta_{\mathrm{eff}}$. Takes 5 minutes, silences a pedantic referee. |
| **Remove MPL from Supp** | ✅ Fully addressed | Correct call — MPL is the wrong model for conditional-state data. |
| **Gauge averaging resolved** | ✅ Fully addressed | Spot-check std $< 2\%$ justifies fixed gauge. |

**Bottom line:** 5 of 6 addendum items are closed. The **lambda sweep / crossover diagram regeneration** is the one remaining critical action.

---

## (b) Anything new that would block submission?

Three items, ranked by severity:

### 🔴 1. Fig. 2b & Fig. 3b provenance (blocking if not fixed)

Embedded `auto_scale` spot check at `chain_strength=1.0` shows:

| $\lambda$ | `auto_scale=True` | `auto_scale=False` |
|-----------|-------------------|--------------------|
| 0.15 | 0.995 | 0.040 |
| 0.20 | 0.988 | 0.006 |
| 0.30 | 0.038 | 0.000 |

The paper's Fig. 2b reports $\mathcal{M} = 0.53$ at $\lambda = 0.2$. If that point came from `auto_scale=True` + `uniform_torque_compensation`, it does **not** describe the nominal Hamiltonian. The true nominal value is $\mathcal{M} \approx 0.006$.

**What is needed:** Either
- (a) Regenerate Fig. 2b and Fig. 3b with `auto_scale=False`, or
- (b) Add a frank Methods footnote: *"Early Phase 3/4 scans used the SDK-default `auto_scale=True`. Spot checks (Supplementary Sec. X) confirm that the $|E|$ and $W$ sweeps are unaffected at $\lambda=0.5$, but the lambda crossover location shifts by $\Delta\lambda\approx0.05$ when `auto_scale` is disabled. The Phase 7 mixed-frustration data and all thermal-marginal diagnostics were submitted with `auto_scale=False`."*

Option (a) is safer for Nature. Option (b) might pass if the rest of the paper is strong, but it gives a referee an easy target.

### 🟡 2. Native-graph frozen state at $|E|=20$, $\lambda=0.5$ (needs one sentence)

Native spot check (`auto_scale=False`, $|E|=20$, $\lambda=0.5$) gives **$\mathcal{M} = 0.94$** — nearly complete memory retention. Yet the embedded main sweep at the same $\lambda=0.5$, $|E|=50$ shows $\mathcal{M} \approx 0.0$.

A referee will ask: *"If this is universal subsystem relaxation, why does a native Zephyr subgraph with 20 environment qubits freeze, while an embedded 3-regular graph with 50 environment qubits relaxes?"*

**Fix:** Add one sentence to the Discussion or Supplementary acknowledging that native and embedded geometries have different effective boundary connectivities, and that the $|E|$ threshold for relaxation is geometry-dependent. Cite the native-graph supplementary table as evidence that relaxation *does* occur on native graphs at larger $|E|$ or stronger $\lambda$.

### 🟢 3. Missing Lindblad unit tests (non-blocking, but cheap)

`lindblad.py` still has zero test coverage. Add a 20-line test: single-qubit relaxation reproduces detailed balance, two-qubit ferromagnet relaxes to correct thermal bias. This is a 10-minute fix that prevents a code-review referee from nagging you.

---

## (c) One-line rating 0–10

**8/10 as-is. 9/10 if Fig. 2b & 3b are regenerated with `auto_scale=False`.**

What would make it a 10/10:
- Regenerate the two affected figures.
- Add the 5-minute single-qubit probe consistency check ($h = 0.25$ vs $0.5$).
- One sentence explaining the native-graph $|E|$-threshold shift.

What would drop it to 6/10:
- Submitting with Fig. 2b / 3b still generated under `auto_scale=True` and no explicit disclaimer. A D-Wave-literate referee will spot this immediately.

---

## Final Word

The science is solid, the controls are now rigorous, and the code is clean. The user caught the `auto_scale` issue before submission, which is exactly when you want to find it. Close that last figure gap and send it.
