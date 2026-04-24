# Decision Memo: Quantum Annealer Self-Thermalization

## Title

**A programmable quantum annealer thermalizes its own subsystems**

## One-Sentence Claim

Above a tunable size/coupling threshold, a programmable quantum annealer acts as its own bath and drives subsystem thermalization; quenched disorder arrests this process and preserves memory.

## Three Diagnostics

1. **Initial-state independence.** The marginal distribution P_S over a small subsystem S converges to the same distribution regardless of the initial state of S, as the environment size |E| and coupling strength lambda increase.

2. **Gibbs-fit quality.** The converged P_S is well described by a single effective Gibbs distribution exp(-beta_eff H_S)/Z, with beta_eff consistent across initial states and stable as |E| grows.

3. **Disorder arrest.** Both diagnostics fail when quenched disorder W exceeds a threshold: memory is preserved and P_S remains initial-state dependent.

## Classical Kill Sentence

If Glauber dynamics at the device temperature T_device quantitatively reproduces all three diagnostics with the same parameter dependences (size, coupling, disorder), the quantum claim is dead.

## Figure Sketch

**Figure 1 — Protocol and concept.**
(a) Reverse-anneal schedule: s=1 → s_p → hold t_p → s=1.
(b) QPU topology with subsystem S highlighted, environment E surrounding it, boundary couplers scaled by lambda.
(c) Cartoon: three different initial states of S all converge to the same P_S after reverse anneal with large E.

**Figure 2 — Global memory loss (discovery phase).**
(a) Total variation distance between P(final|init_1) and P(final|init_2) vs pause time t_p at several pause depths s_p. Curves decay toward zero.
(b) Gibbs-fit chi-squared vs t_p — improves in the same regime.
(c) Same as (a) for several system sizes N — larger systems thermalize more completely.

**Figure 3 — Subsystem thermalization (the Nature figure).**
(a) P_S for different initial states at several |E| values — distributions converge as |E| grows.
(b) Initial-state TVD vs |E| at fixed lambda — clear monotonic decrease.
(c) TVD vs lambda at fixed |E| — same trend.
(d) Extracted beta_eff from Gibbs fits — consistent across initial states and stable with |E|.

**Figure 4 — Disorder arrests thermalization.**
(a) Initial-state TVD vs disorder strength W — increases with W.
(b) Crossover diagram in the (lambda or |E|, W) plane — thermalized vs memory-preserving regions.
(c) Same crossover on both QPU platforms in dimensionless units — universality.

**Figure 5 — Classical baselines fail.**
(a) Annealer data vs Glauber dynamics — quantitative mismatch in crossover location or thermalization timescale.
(b) Annealer vs classical spin-vector MC — different functional form.
(c) Exact diagonalization (small system) agrees with annealer — confirming quantum coherent origin.
(d) Optional: Lindblad comparison showing Markovian bath insufficient.
