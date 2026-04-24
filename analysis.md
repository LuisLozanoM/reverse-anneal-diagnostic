# Structured Analysis: Quantum Annealer Self-Thermalization

This document provides a thorough 10-section analysis of the paper plan for a Nature/Nature Physics submission on self-thermalization in programmable quantum annealers.

---

## 1. Core Scientific Claim

### Flagship Claim (One Sentence)

> Above a tunable size/coupling threshold, a programmable quantum annealer acts as its own bath and drives subsystem thermalization; quenched disorder arrests this process and preserves memory.

This is a two-part claim with complementary physics:

1. **Thermalization half:** A small subsystem S, when coupled to a sufficiently large environment E drawn from the same annealer, loses memory of its initial state and relaxes to an effective Gibbs distribution. The coupling strength lambda and environment size |E| serve as tunable control knobs.

2. **Disorder/memory half:** Introducing quenched disorder (random fields of strength W) arrests this thermalization process. The subsystem retains memory of its preparation, analogous to many-body localization phenomenology.

### Upgraded Concept (Simpler → Stronger)

The original idea was a global-memory experiment: prepare different initial configurations of the full system, reverse anneal, and check whether the final distribution depends on the initial state. The upgraded version partitions the system into a small subsystem S and a controllable environment E, making the claim much stronger:

- We can vary |E| and lambda independently
- We can ask whether P_S (the reduced distribution over S) becomes initial-state independent
- We can check consistency with a single effective Gibbs description of S
- The environment is programmable, not assumed — the annealer literally builds its own bath

This upgrade transforms the experiment from "the annealer forgets" (interesting but hard to interpret) into "the annealer thermalizes a subsystem via a controllable environment" (fundamental physics, testable against ETH predictions).

### Three Diagnostics

1. **Initial-state independence:** P_S converges to the same distribution regardless of how S was prepared
2. **Gibbs-fit quality:** The converged P_S is well described by exp(-beta_eff H_S) with a single effective inverse temperature beta_eff
3. **Disorder arrest:** Both diagnostics fail (memory preserved) when quenched disorder W exceeds a threshold

### Classical Kill Sentence

If a purely classical model (Glauber dynamics, classical spin-vector Monte Carlo) at the known device temperature reproduces all three diagnostics with the same parameter dependences, the quantum claim is dead. The experiment must show features that classical thermal models cannot replicate — either in the thermalization timescales, the functional form of the crossover, or the disorder threshold.

---

## 2. Key Phases

### Phase 1: Define the Nature Sentence

**Goal:** One-page decision memo containing title, one-sentence claim, three diagnostics, one classical kill sentence, and a four-figure sketch.

**Deliverables:**
- Working title
- The flagship sentence (finalized wording)
- Three measurable diagnostics with quantitative criteria
- A single sentence defining what classical result would kill the paper
- Sketch of four main figures (hand-drawn or schematic level)

**Go/No-Go:** The memo must be internally consistent. If we cannot write a coherent one-sentence claim that distinguishes this from classical thermalization, stop.

### Phase 2: Build Proof Architecture

**Goal:** Lock down all experimental and theoretical ingredients before touching the QPU.

**Deliverables:**
- Hamiltonian family: which couplers, which topology, which disorder distribution
- Observable definitions: exact formulas for initial-state distance, Gibbs fit chi-squared, memory order parameter
- Fitting procedure: maximum likelihood vs least squares for Gibbs fits, bootstrap error bars
- Classical baselines: code for exact diag, Lindblad, spin-vector MC, Glauber
- Metadata schema: what gets logged per reverse-anneal job (anneal schedule, initial state, timing, QPU ID, calibration epoch)

**Hardware Lock:**
| Property | Advantage2 (Primary) | Advantage_system6.4 (Secondary) |
|----------|----------------------|----------------------------------|
| Qubits | 4,579 | 5,612 |
| Energy scale | 2.308 GHz | 1.281 GHz |
| Effective temperature | 0.112 | 0.221 |
| Role | Primary data | Cross-validation |

**Go/No-Go:** All code must produce correct results on toy Hamiltonians before proceeding. Metadata schema must be finalized. If energy scales differ too much between QPUs for meaningful comparison, document and flag.

### Phase 3: Run Easy Discovery Version

**Goal:** First experimental pass — reverse annealing with multiple initial states.

**Protocol:**
1. Prepare two or more distinct initial states (e.g., all-up, all-down, random, domain wall)
2. Reverse anneal to pause point s_p
3. Hold for time t_p
4. Return-ramp to s = 1 and read out
5. Repeat for many shots
6. Scan over N (system size), s_p (pause depth), t_p (pause duration), W (disorder), and return-ramp shape

**What to look for:**
- Initial-state dependence decreases with increasing t_p and decreasing s_p (deeper pause → more thermalization)
- Gibbs-fit quality improves in the same regime
- Adding disorder W preserves initial-state memory
- Results are reproducible across calibration cycles

**Go/No-Go:** If initial-state dependence does not decrease at all — even at the deepest pause and longest hold — the mechanism is not there. Kill the project or pivot. If Gibbs fits are always bad, the effective-temperature picture may not apply.

### Phase 4: Upgrade to Subsystem-Bath Experiment

**Goal:** The Nature-level upgrade — partition the system into S (subsystem) and E (environment).

**Protocol:**
1. Choose a small subsystem S (e.g., 4-8 qubits in a connected subgraph)
2. Couple S to environment E via programmable couplers with strength lambda
3. Vary |E| (number of environment qubits) while keeping S fixed
4. Vary lambda (S-E coupling strength)
5. Optionally vary W_E (disorder in the environment only, or in S only, or both)
6. For each setting, prepare different initial states of S (with E fixed or also varied)
7. Reverse anneal, hold, return, read out
8. Compute P_S (marginal distribution over S) for each initial state

**Key quantity:** P_S becomes initial-state independent AND consistent with a single effective Gibbs description as |E| and lambda increase. This is the thermalization signature.

**Go/No-Go:** If P_S never converges across initial states, subsystem thermalization is not occurring. If P_S converges but is not Gibbs-like, the physics is different (maybe interesting, but not the claimed thermalization). If |E| dependence is flat, the environment is not acting as a bath.

### Phase 5: Kill Classical Explanations

**Goal:** Demonstrate that classical models cannot reproduce the observed thermalization phenomenology.

**Classical baselines:**
1. **Exact diagonalization (ED):** Full quantum simulation of S+E for small sizes. Benchmark: does the closed quantum system show the same thermalization? (It should, via ETH.)
2. **Lindblad master equation:** Open quantum system with a phenomenological bath. Benchmark: can a Markovian classical bath reproduce the data? If yes, quantum coherence is not essential.
3. **Classical spin-vector Monte Carlo:** Replace quantum spins with classical vectors. Benchmark: does classical thermal physics suffice?
4. **Glauber dynamics:** Classical stochastic spin flips at the device temperature. Benchmark: is the annealer just doing classical Metropolis sampling?

**Hardware control sweeps:**
- Vary anneal time (changes the effective quantum-to-classical crossover)
- Vary pause point s_p (controls the transverse field and hence the quantum character)
- Compare Advantage2 and Advantage_system6.4 (different energy scales and temperatures)

**Go/No-Go:** If Glauber dynamics at the device temperature quantitatively reproduces all features, the paper is dead for Nature. We need at least one observable where the quantum annealer deviates from all classical baselines.

### Phase 6: Universal Physics Framing

**Goal:** Frame the results as universal condensed matter physics, not a hardware demo.

**Required elements:**
- **Order parameter:** A scalar quantity that distinguishes thermalized from non-thermalized (e.g., initial-state distance or Gibbs-fit residual)
- **Control parameter:** What drives the transition (|E|, lambda, W, or a combination)
- **Crossover line:** In the (control parameter, disorder) plane, map where thermalization breaks down
- **Physical interpretation:** Connect to eigenstate thermalization hypothesis (ETH), many-body localization (MBL), or thermalization in isolated quantum systems

**Key framing principle:** "The hardware is the platform, not the phenomenon." We are using D-Wave as a programmable quantum simulator to study thermalization physics, not reporting on D-Wave engineering. The physics must be universal — it should apply to any system with the same Hamiltonian, not just to this specific QPU.

**Go/No-Go:** If we cannot identify a clean order parameter and control parameter, the story is too messy for Nature. If the crossover is device-specific (depends on calibration details rather than Hamiltonian parameters), the universality claim fails.

### Phase 7: Write for Editors

**Goal:** Produce a manuscript that passes the Nature editorial desk.

**Strategy:**
1. Write the 150-200 word summary first (this IS the paper in compressed form)
2. Build the figure storyboard before writing text
3. Main text answers five questions:
   - **Big question:** How does thermalization emerge in isolated quantum systems?
   - **What's missing:** No experimental platform has demonstrated tunable subsystem thermalization with programmable disorder in a many-body quantum system.
   - **What's new:** We show that a quantum annealer provides exactly this platform.
   - **Central result:** Subsystem thermalization with environment-size and coupling control, arrested by disorder.
   - **Why it matters beyond hardware:** Direct experimental probe of ETH/MBL phenomenology at unprecedented scale.

**Go/No-Go:** If the 150-word summary does not excite a non-expert physicist, iterate until it does or accept that the story is not Nature-level.

### Phase 8: Submission Strategy

**Decision tree:**

**Nature** (if ALL 5 criteria are met):
1. Subsystem thermalization clearly demonstrated with |E| and lambda control
2. Disorder arrest is clean and reproducible
3. At least one classical baseline fails to reproduce the data
4. Universal physics framing with clean order/control parameters
5. Cross-validation on two QPU platforms

**Nature Physics** (if criteria 1-2 are met but 3-5 are partial):
- Thermalization is clear but classical kill is ambiguous
- Or: only one QPU platform
- Or: framing is solid but crossover is not perfectly clean

**Nature Communications** (if criteria 1-2 are met but 3-5 are weak):
- Phenomenology is interesting but universality is limited

**Communications Physics / PRL** (if only criterion 1 is met):
- Subsystem thermalization is demonstrated but the story is primarily a hardware result

---

## 3. Hardware Requirements

### QPU Specifications

| Property | Advantage2 (Primary) | Advantage_system6.4 (Secondary) |
|----------|----------------------|----------------------------------|
| Topology | Zephyr | Pegasus |
| Working qubits | 4,579 | 5,612 |
| Energy scale | 2.308 GHz | 1.281 GHz |
| Effective temperature (kT/h) | 0.112 | 0.221 |
| Ratio energy/temp | ~20.6 | ~5.8 |
| Role in paper | Primary experimental platform | Cross-validation and universality |

### Cross-Platform Matching Requirements

To claim universality (criterion 5 for Nature), we must demonstrate that the thermalization phenomenology depends on Hamiltonian parameters (N, lambda, W), not on hardware-specific details. This requires:

1. **Same Hamiltonian family** embedded on both QPUs (may require restricting to subgraphs common to Zephyr and Pegasus, or using minor embedding)
2. **Matched dimensionless parameters:** Because energy scales and temperatures differ, we should plot results against dimensionless ratios (e.g., J/kT, W/J, lambda/J) rather than raw parameter values
3. **Consistent anneal schedules:** The s_p values should correspond to the same transverse-field-to-coupling ratio, not the same raw s value (since the anneal schedule A(s)/B(s) differs between QPUs)
4. **Matched disorder realizations:** Use the same random seeds for disorder configurations, embedded appropriately on each topology

### Why Two QPUs Are Essential

The Advantage2 has a higher energy-to-temperature ratio (~20.6 vs ~5.8), meaning it operates in a more quantum regime. If thermalization signatures appear on both platforms with the same Hamiltonian-parameter dependence, the effect is robust. If the Advantage2 shows stronger quantum features (as expected from its higher energy scale), this strengthens the quantum interpretation.

### Classical Hardware

M3 Max with 128 GB RAM is sufficient for:
- Exact diagonalization up to ~20 qubits (Hilbert space dimension 2^20 ~ 10^6)
- Lindblad master equation for ~12-15 qubits
- Classical spin-vector and Glauber Monte Carlo for any size
- All data analysis and fitting

Use MPS backend for any PyTorch-based calculations.

---

## 4. Figure Architecture

### Figure 1: Setup and Protocol

**Content:** Schematic of the reverse-anneal protocol and the subsystem-environment partition.

- Panel (a): Anneal schedule showing forward anneal, reverse to s_p, hold for t_p, return
- Panel (b): Schematic of S (highlighted) coupled to E on the QPU topology
- Panel (c): Cartoon showing different initial states of S and the expected convergence

**Purpose:** Orients the reader. Makes the protocol immediately clear.

### Figure 2: Thermalization of the Full System (Discovery Phase)

**Content:** Evidence that the full system loses initial-state memory.

- Panel (a): Distribution distance (e.g., total variation distance between P(final | init_1) and P(final | init_2)) vs t_p for several s_p values
- Panel (b): Gibbs-fit quality (chi-squared or KL divergence) vs t_p
- Panel (c): Same as (a) but for several system sizes N

**Purpose:** Establishes the basic phenomenon before the upgrade.

### Figure 3: Subsystem Thermalization (The Nature Figure)

**Content:** The central result — subsystem S thermalizes as |E| and lambda increase.

- Panel (a): P_S for different initial states, shown at several |E| values — convergence as |E| grows
- Panel (b): Initial-state distance for S vs |E| at fixed lambda — clear decrease
- Panel (c): Same as (b) but vs lambda at fixed |E|
- Panel (d): Effective temperature beta_eff extracted from Gibbs fits — consistent across initial states and stable as |E| grows

**Purpose:** This is the figure that justifies Nature. Shows programmable, tunable subsystem thermalization.

### Figure 4: Disorder Arrests Thermalization

**Content:** Quenched disorder preserves memory.

- Panel (a): Initial-state distance vs W at fixed |E| and lambda — increases with W
- Panel (b): Crossover diagram in the (lambda or |E|, W) plane — thermalized vs memory-preserving regions
- Panel (c): Same crossover on both QPUs — universality

**Purpose:** Completes the two-part claim. Connects to MBL phenomenology.

### Figure 5: Classical Baselines Fail

**Content:** Classical models cannot reproduce the quantum annealer data.

- Panel (a): Comparison of annealer data vs Glauber dynamics — quantitative mismatch
- Panel (b): Comparison vs classical spin-vector MC — different crossover location or shape
- Panel (c): Exact diagonalization for small S+E — agreement with annealer, confirming quantum coherent origin
- Panel (d): [OPTIONAL] Lindblad comparison showing Markovian bath is insufficient

**Purpose:** Kills the classical interpretation. Distinguishes this from a fancy random number generator.

---

## 5. Journal Strategy

### The Nature Criteria (All 5 Required)

1. **Tunable subsystem thermalization:** P_S converges across initial states as |E| and lambda increase, with clear functional dependence
2. **Disorder arrest:** Clean, reproducible breakdown of thermalization with increasing W
3. **Classical baselines fail:** At least one classical model quantitatively fails to reproduce the data
4. **Universal physics framing:** Clean order parameter, control parameter, and crossover line; connection to ETH/MBL
5. **Cross-QPU validation:** Consistent results on Advantage2 and Advantage_system6.4 when plotted against dimensionless Hamiltonian parameters

### Submission Ladder

| Target | When | Condition |
|--------|------|-----------|
| **Nature** | All 5 criteria met | Full subsystem thermalization + disorder arrest + classical kill + universal framing + two QPUs |
| **Nature Physics** | Criteria 1-2 met, 3-5 partial | Thermalization clear, classical kill ambiguous OR single QPU OR imperfect crossover |
| **Nature Communications** | Criteria 1-2 met, 3-5 weak | Interesting phenomenology, limited universality |
| **Comm. Phys. / PRL** | Only criterion 1 met | Subsystem thermalization demonstrated, primarily a hardware result |

### Editorial Positioning

For Nature editors, the paper must answer: "Why should a physicist who has never used a quantum annealer care?" The answer is: this is the first experimental platform that provides tunable, programmable access to subsystem thermalization and its disorder-induced arrest in a many-body quantum system at scale.

---

## 6. Classical Baselines Needed

### Exact Diagonalization (ED)

**Purpose:** Ground truth for the quantum dynamics of S+E at small sizes.

- Full diagonalization of H_{S+E} in the computational basis
- Compute time evolution under the instantaneous Hamiltonian at the pause point
- Extract P_S by tracing out E
- **Limit:** ~20 qubits on M3 Max (128 GB), ~24-26 with careful memory management
- **Role in paper:** If ED agrees with annealer data for small systems, the annealer is correctly simulating quantum thermalization. This supports the quantum interpretation.

### Lindblad Master Equation

**Purpose:** Test whether a Markovian (memoryless) classical bath can explain the data.

- Model S+E as an open system coupled to a Markovian environment
- Solve the Lindblad equation d rho/dt = -i[H, rho] + sum_k (L_k rho L_k^dag - 1/2 {L_k^dag L_k, rho})
- Extract P_S from the steady state or time-evolved state
- **Limit:** ~12-15 qubits (density matrix is 2^N x 2^N)
- **Role in paper:** If Lindblad reproduces the data, the thermalization is due to an external classical bath (e.g., phonons in the QPU), not self-thermalization. The claim "acts as its own bath" requires that Lindblad with a phenomenological bath does NOT fully explain the phenomenology.

### Classical Spin-Vector Monte Carlo

**Purpose:** Test whether classical thermal physics (no quantum coherence) suffices.

- Replace each qubit with a classical spin vector (or Ising variable)
- Run Monte Carlo at the device temperature
- Compute the same observables (initial-state distance, Gibbs fit)
- **Limit:** Any system size (polynomial scaling)
- **Role in paper:** If classical MC reproduces the data, there is no quantum advantage to the thermalization mechanism. The crossover or threshold must differ between quantum and classical.

### Glauber Dynamics

**Purpose:** The most dangerous classical baseline — stochastic single-spin-flip dynamics at the device temperature.

- At each step, flip spin i with probability p = 1/(1 + exp(Delta E / kT))
- This is exactly what the annealer would do if it were a classical thermal machine
- Compute thermalization time, initial-state distance decay, and Gibbs-fit quality
- **Limit:** Any system size
- **Role in paper:** If Glauber dynamics at T = T_device reproduces all three diagnostics, the annealer is just doing classical thermal sampling and the paper is dead. This is the classical kill sentence incarnated.

### Baseline Comparison Strategy

We need a hierarchy of explanatory power:

```
Glauber (classical stochastic) ⊂ Spin-vector MC (classical thermal) ⊂ Lindblad (open quantum, Markovian) ⊂ ED (closed quantum, unitary)
```

If the annealer data matches ED but not the classical baselines, the thermalization is quantum. If it matches Lindblad but not Glauber, the quantum bath is non-Markovian. If it matches Glauber, it is classical.

---

## 7. Key Experimental Parameters

| Parameter | Symbol | Description | Typical Range | Role |
|-----------|--------|-------------|---------------|------|
| Pause point | s_p | Position in anneal schedule where we hold (s=0 is full transverse field, s=1 is fully classical) | 0.3 - 0.7 | Controls the ratio of transverse field to Ising coupling at the hold point. Lower s_p = more quantum. |
| Pause time | t_p | Duration of the hold at s_p | 1 - 1000 μs | Controls how long the system evolves at the pause Hamiltonian. Longer = more thermalization opportunity. |
| System size | N | Total number of qubits (S + E) | 8 - 200+ | More qubits = larger Hilbert space = more effective bath. |
| Disorder strength | W | Magnitude of random longitudinal fields h_i drawn from [-W, W] | 0 - 2.0 (in units of J) | Breaks translational symmetry. Expected to arrest thermalization above a threshold. |
| S-E coupling | lambda | Strength of couplers connecting subsystem S to environment E | 0 - 1.0 (in units of max coupler strength) | Controls the rate of energy/information exchange between S and E. |
| Environment size | \|E\| | Number of qubits in the environment partition | 0 - N-\|S\| | The effective bath size. Thermalization should improve with larger \|E\|. |
| Subsystem size | \|S\| | Number of qubits in the subsystem partition | 4 - 8 | Must be small enough for P_S to be measurable with reasonable statistics. |

### Scan Strategy

**Primary scans (Phase 3):**
- t_p at fixed s_p, N, W=0
- s_p at fixed t_p, N, W=0
- N at fixed s_p, t_p, W=0
- W at fixed s_p, t_p, N

**Upgrade scans (Phase 4):**
- |E| at fixed |S|, lambda, W=0
- lambda at fixed |S|, |E|, W=0
- W at fixed |S|, |E|, lambda (the disorder arrest scan)
- Combined (|E|, W) grid for the crossover diagram

---

## 8. Critical D-Wave API Details

### Reverse Annealing Interface

Reverse annealing on D-Wave requires three key parameters:

1. **`initial_state`**: A dictionary mapping qubit indices to classical spin values (+1 or -1). This defines the starting configuration for the reverse anneal.
   ```python
   initial_state = {0: 1, 1: -1, 2: 1, ...}  # one entry per active qubit
   ```

2. **`anneal_schedule`**: A list of (time, s) pairs defining the piecewise-linear anneal trajectory. For reverse annealing, the schedule starts at s=1, decreases to s_p, optionally holds, then returns to s=1.
   ```python
   anneal_schedule = [
       (0.0, 1.0),           # start at s=1 (classical)
       (t_ramp_down, s_p),   # ramp down to pause point
       (t_ramp_down + t_p, s_p),  # hold at pause point
       (t_total, 1.0)        # ramp back to s=1
   ]
   ```

3. **`reinitialize_state`**: Boolean flag.
   - `True`: reinitialize to `initial_state` before each anneal (each shot starts from the same state)
   - `False`: use the result of the previous anneal as the starting state for the next one (iterated reverse annealing)
   - **For this experiment, use `True`** — we want each shot to start from the specified initial state to measure the initial-state dependence.

### Timing Controls

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `annealing_time` | Total anneal time (not used when `anneal_schedule` is specified) | N/A |
| `t_ramp_down` | Time to ramp from s=1 to s_p | 1 - 10 μs |
| `t_p` (hold) | Duration at s_p | 1 - 1000 μs |
| `t_ramp_up` | Time to ramp from s_p back to s=1 | 1 - 10 μs |
| `num_reads` | Number of shots per job | 100 - 10,000 |
| `auto_scale` | Whether to auto-scale h and J to the hardware range | True (usually) |

### Platform-Specific Constraints

**Advantage2 (primary):**
- Maximum anneal time: [TO BE DETERMINED — check current solver properties]
- Minimum pause: [TO BE DETERMINED]
- Anneal schedule resolution: [TO BE DETERMINED]
- Energy scale factor: 2.308 GHz → stronger couplings, better signal-to-noise at the pause point

**Advantage_system6.4 (secondary):**
- Maximum anneal time: [TO BE DETERMINED]
- Minimum pause: [TO BE DETERMINED]
- Anneal schedule resolution: [TO BE DETERMINED]
- Energy scale factor: 1.281 GHz → weaker couplings, higher effective temperature

### Sample API Call

```python
from dwave.system import DWaveSampler, EmbeddingComposite

sampler = DWaveSampler(solver='Advantage2_prototype2.6')

# Define Hamiltonian
h = {i: 0.0 for i in range(N)}  # longitudinal fields
J = {(i, j): -1.0 for i, j in edges}  # couplings

# Define initial state
initial_state = {i: 1 for i in range(N)}  # all-up

# Define anneal schedule (reverse anneal)
s_p = 0.4
t_ramp = 5.0  # μs
t_hold = 100.0  # μs
anneal_schedule = [
    (0.0, 1.0),
    (t_ramp, s_p),
    (t_ramp + t_hold, s_p),
    (2 * t_ramp + t_hold, 1.0)
]

# Run reverse anneal
response = sampler.sample_ising(
    h, J,
    initial_state=initial_state,
    anneal_schedule=anneal_schedule,
    reinitialize_state=True,
    num_reads=1000
)
```

### Critical Notes

- The `anneal_schedule` replaces `annealing_time` — do not set both
- Times are in microseconds
- The schedule must be piecewise linear and monotonic in time
- s must start and end at 1.0 for reverse annealing
- s_p must be within the allowed range for the solver (typically 0.0 to 1.0)
- `initial_state` must include all qubits used in the problem
- Check `solver.properties['max_anneal_schedule_points']` for the maximum number of (time, s) pairs

---

## 9. Risks and Kill Criteria

### Phase 1 Risks
- **Risk:** Cannot articulate a single coherent Nature-level sentence
- **Kill criterion:** If after one week of iteration the claim is still muddled, the concept is not sharp enough

### Phase 2 Risks
- **Risk:** Advantage2 availability or calibration instability
- **Kill criterion:** If the primary QPU is unavailable for >2 weeks or calibration drift exceeds acceptable levels, switch to Advantage_system6.4 as primary
- **Risk:** Classical baseline code produces wrong results on benchmarks
- **Kill criterion:** Must validate against known exact results before using

### Phase 3 Risks
- **Risk:** No initial-state dependence decrease observed at any (s_p, t_p)
- **Kill criterion:** If scanning the full (s_p, t_p, N) space shows no trend, the mechanism is absent. Kill the project.
- **Risk:** Gibbs fits are never good — the effective temperature picture does not apply
- **Kill criterion:** If chi-squared is always large, consider alternative descriptions or kill the Gibbs framing
- **Risk:** Results are not reproducible across calibration cycles
- **Kill criterion:** If the same parameters give qualitatively different results on different days, the signal is calibration-dependent, not physics

### Phase 4 Risks
- **Risk:** P_S does not converge across initial states even with large |E|
- **Kill criterion:** If convergence is absent for |E| up to the maximum practical size, subsystem thermalization is not occurring
- **Risk:** |E| dependence is flat — environment size does not matter
- **Kill criterion:** If thermalization does not improve with |E|, the environment is not acting as a bath
- **Risk:** lambda dependence is too weak or non-monotonic
- **Kill criterion:** Must show clear, monotonic improvement in thermalization with lambda

### Phase 5 Risks
- **Risk:** Glauber dynamics reproduces all features at T = T_device
- **Kill criterion:** Paper is dead for Nature. May still be publishable at a lower venue as a hardware characterization, but the quantum claim is gone.
- **Risk:** Classical spin-vector MC reproduces the crossover
- **Kill criterion:** The quantum interpretation is substantially weakened
- **Risk:** ED disagrees with annealer data at small sizes
- **Kill criterion:** If the quantum ground truth does not match, something is wrong with the experiment or the interpretation

### Phase 6 Risks
- **Risk:** No clean order parameter exists
- **Kill criterion:** If the thermalization-to-memory transition is not captured by a single scalar, the story is too complex for Nature
- **Risk:** Crossover depends on hardware-specific details (calibration, embedding)
- **Kill criterion:** Universality claim fails. Downgrade to Nature Physics or lower

### Phase 7 Risks
- **Risk:** 150-word summary is not compelling
- **Kill criterion:** If a non-expert physicist reads it and asks "so what?", iterate

### Phase 8 Risks
- **Risk:** Referee demands impossible controls (e.g., decoherence-free subspace, full quantum state tomography)
- **Mitigation:** Preemptively address in supplementary; frame as analog quantum simulation, not digital quantum computation
- **Risk:** Competing group publishes similar results first
- **Mitigation:** Speed. The D-Wave platform access and the subsystem-environment framing are our differentiators

---

## 10. What Makes This Nature vs Nature Physics

### The Core Distinction

**Nature Physics paper (good but not enough):**
- "A quantum annealer loses memory of its initial state when reverse annealed" — interesting hardware observation
- Global memory effect without subsystem resolution
- Comparison to one or two classical baselines
- Results on one QPU
- Framed as "what the annealer does"

**Nature paper (the upgrade):**
- "A quantum annealer acts as its own bath and thermalizes a programmable subsystem" — fundamental physics result
- Subsystem S with tunable environment E — direct probe of thermalization mechanism
- All four classical baselines compared, with at least one clearly failing
- Cross-validated on two QPUs with different energy scales/temperatures
- Framed as "what thermalization looks like in a controllable quantum many-body system"
- Explicit connection to ETH and MBL — the annealer becomes an experimental platform for studying these fundamental questions at unprecedented scale

### The Five Upgrades (Concretely)

1. **Subsystem resolution:** Measuring P_S instead of P_full transforms "memory loss" into "thermalization"
2. **Tunable environment:** Varying |E| and lambda proves the mechanism (it is the environment causing thermalization, not something else)
3. **Disorder control:** W is the control knob that arrests thermalization, connecting to MBL phenomenology
4. **Classical baselines killed:** At least one classical model must fail, otherwise this is just fancy Monte Carlo
5. **Two QPU platforms:** Universality — the physics depends on the Hamiltonian, not the hardware

### What Editors Will Ask

A Nature editor will ask three questions:

1. "Is this physics or engineering?" → The universal framing (Phase 6) answers this. The annealer is a platform for studying thermalization physics, not a subject of engineering characterization.

2. "Could this be done classically?" → Phase 5 answers this. If yes, the paper is not Nature.

3. "Why should the broad physics community care?" → The connection to ETH/MBL answers this. Thermalization in isolated quantum systems is one of the central questions in quantum statistical mechanics. This provides a new experimental platform at scale.

