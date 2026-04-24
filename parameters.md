# Experimental Parameters Reference

Complete reference for all experimental parameters in the locth1 self-thermalization project.

---

## Primary Experimental Parameters

| Parameter | Symbol | Type | Description | Range | Units | Role |
|-----------|--------|------|-------------|-------|-------|------|
| Pause point | s_p | Anneal control | Position in the anneal schedule where the system is held. s=0 is full transverse field (quantum), s=1 is fully Ising (classical). | 0.3 – 0.7 | dimensionless | Controls the ratio of transverse field to Ising coupling at the hold point. Lower s_p means stronger quantum fluctuations during the pause. |
| Pause time | t_p | Anneal control | Duration of the hold at the pause point s_p. | 1 – 1000 | μs | Determines how long the system evolves under the pause Hamiltonian H(s_p). Longer t_p gives more time for thermalization. |
| System size | N | Topology | Total number of active qubits (|S| + |E|). | 8 – 200+ | qubits | Larger N means a larger Hilbert space and a more effective environment. |
| Subsystem size | \|S\| | Partition | Number of qubits in the subsystem S whose thermalization we measure. | 4 – 8 | qubits | Must be small enough that P_S can be estimated with reasonable shot statistics. 2^|S| states to resolve. |
| Environment size | \|E\| | Partition | Number of qubits in the environment E (= N - \|S\|). | 0 – N-\|S\| | qubits | The effective bath size. Thermalization of S should improve as \|E\| grows. |
| S-E coupling | lambda | Hamiltonian | Strength of couplers connecting S to E, as a fraction of maximum coupler strength. | 0.0 – 1.0 | dimensionless | Controls the rate of energy and information exchange between S and E. lambda=0 means S is decoupled. |
| Disorder strength | W | Hamiltonian | Magnitude of random longitudinal fields h_i drawn uniformly from [-W, W]. | 0.0 – 2.0 | units of J | Breaks translational symmetry. Expected to arrest thermalization above a critical W_c. |
| Disorder in E only | W_E | Hamiltonian | Disorder applied only to environment qubits. | 0.0 – 2.0 | units of J | Allows independent control of disorder in S and E. |

---

## Anneal Schedule Parameters

| Parameter | Symbol | Description | Typical Values | Units |
|-----------|--------|-------------|----------------|-------|
| Ramp-down time | t_ramp_down | Time to ramp from s=1 to s_p | 1 – 10 | μs |
| Hold time | t_hold | Duration at s_p (same as t_p) | 1 – 1000 | μs |
| Ramp-up time | t_ramp_up | Time to ramp from s_p back to s=1 | 1 – 10 | μs |
| Total anneal time | t_total | t_ramp_down + t_hold + t_ramp_up | 3 – 1020 | μs |
| Number of schedule points | — | Points in the piecewise-linear schedule | 4 (typical) | — |

### Standard Reverse-Anneal Schedule

```
s
1.0  ●────────────────────────────●
     │                            │
     │                            │
s_p  │    ●────────────────●      │
     │   ╱                  ╲     │
     │  ╱                    ╲    │
     └──────────────────────────── t
     0  t_rd   t_rd+t_h   t_total
```

Where:
- t_rd = t_ramp_down
- t_h = t_hold = t_p
- t_total = t_rd + t_h + t_ramp_up

---

## Sampling Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| num_reads | Number of shots per job | 1,000 – 10,000 |
| num_initial_states | Number of different initial states to compare | 2 – 6 |
| num_disorder_realizations | Number of random disorder instances to average over | 10 – 50 |
| reinitialize_state | Whether to reset to initial_state before each shot | True (always for this experiment) |
| auto_scale | Whether to auto-scale h, J to hardware range | True |

---

## Derived Quantities

| Quantity | Symbol | Formula | Description |
|----------|--------|---------|-------------|
| Initial-state distance | D | TVD(P_S\|init_1, P_S\|init_2) | Total variation distance between subsystem distributions from different initial states. D→0 means thermalization. |
| Gibbs-fit residual | G | KL(P_S \|\| P_Gibbs) or chi-squared | Quality of fit to exp(-beta_eff H_S^eff)/Z. G→0 means Gibbs-like. |
| Effective temperature | beta_eff | Fit parameter | Inverse temperature extracted from best Gibbs fit to P_S. |
| Memory order parameter | M | f(D, number of initial states) | Scalar summarizing how much initial-state information survives. [TO BE DETERMINED — exact definition depends on Phase 1 decisions] |
| Dimensionless disorder | w | W / J | Disorder in units of the typical coupling strength. |
| Dimensionless coupling | l | lambda * J / kT | S-E coupling in units of thermal energy. |

---

## Phase 3 Scan Plan

### Scan 1: Pause Time (Thermalization Dynamics)
| Fixed | Variable | Values |
|-------|----------|--------|
| s_p = 0.4, N = 50, W = 0 | t_p | 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 μs |

### Scan 2: Pause Depth (Quantum Character)
| Fixed | Variable | Values |
|-------|----------|--------|
| t_p = 100 μs, N = 50, W = 0 | s_p | 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70 |

### Scan 3: System Size (Finite-Size Effects)
| Fixed | Variable | Values |
|-------|----------|--------|
| s_p = 0.4, t_p = 100 μs, W = 0 | N | 8, 16, 32, 50, 75, 100, 150, 200 |

### Scan 4: Disorder Strength (Memory Preservation)
| Fixed | Variable | Values |
|-------|----------|--------|
| s_p = 0.4, t_p = 100 μs, N = 50 | W | 0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0 |

### Scan 5: Return-Ramp Shape
| Fixed | Variable | Values |
|-------|----------|--------|
| s_p = 0.4, t_p = 100 μs, N = 50, W = 0 | t_ramp_up | 1 (fast), 5 (medium), 20 (slow) μs |

---

## Phase 4 Scan Plan

### Scan A: Environment Size
| Fixed | Variable | Values |
|-------|----------|--------|
| \|S\| = 6, lambda = 0.5, s_p = 0.4, t_p = 100 μs, W = 0 | \|E\| | 4, 8, 16, 32, 50, 75, 100 |

### Scan B: S-E Coupling
| Fixed | Variable | Values |
|-------|----------|--------|
| \|S\| = 6, \|E\| = 50, s_p = 0.4, t_p = 100 μs, W = 0 | lambda | 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 |

### Scan C: Disorder Sweep (Subsystem Version)
| Fixed | Variable | Values |
|-------|----------|--------|
| \|S\| = 6, \|E\| = 50, lambda = 0.5, s_p = 0.4, t_p = 100 μs | W | 0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0 |

### Scan D: Crossover Diagram
| Fixed | Variable | Grid |
|-------|----------|------|
| \|S\| = 6, s_p = 0.4, t_p = 100 μs | (\|E\| or lambda) vs W | ~10 x 10 grid |

---

## QPU-Specific Parameter Notes

### Advantage2
- Higher energy scale (2.308 GHz) → couplings are ~1.8x stronger in absolute terms
- Lower effective temperature (0.112) → better signal-to-noise for thermalization signatures
- J/kT ratio is larger → deeper quantum regime at the same s_p
- Preferred for primary data collection

### Advantage_system6.4
- Lower energy scale (1.281 GHz) → weaker couplings
- Higher effective temperature (0.221) → noisier, more classical
- Use for cross-validation: same dimensionless parameters (W/J, lambda*J/kT) should give same physics
- If results differ at matched dimensionless parameters → hardware-specific effect (bad for universality claim)

