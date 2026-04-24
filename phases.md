# Phase-by-Phase Execution Plan

Detailed checklists and milestones for each of the 8 phases of the locth1 paper project.

---

## Phase 1: Define the Nature Sentence

**Duration estimate:** 1-2 days  
**Dependencies:** None

### Checklist

- [ ] Draft working title (aim for ~10 words, no jargon beyond "quantum annealer" and "thermalization")
- [ ] Write the one-sentence flagship claim
- [ ] Define three quantitative diagnostics:
  - [ ] Diagnostic 1: Initial-state distance metric (total variation distance, KL divergence, or trace distance on P_S)
  - [ ] Diagnostic 2: Gibbs-fit quality metric (chi-squared, KL divergence from best-fit Gibbs)
  - [ ] Diagnostic 3: Memory order parameter (how much initial-state information survives)
- [ ] Write the classical kill sentence (one sentence: "If X, the paper is dead")
- [ ] Sketch four main figures (schematic level, on paper or tablet):
  - [ ] Figure 1: Protocol and setup
  - [ ] Figure 2: Discovery-phase data
  - [ ] Figure 3: Subsystem thermalization (the Nature figure)
  - [ ] Figure 4: Disorder arrest + crossover
- [ ] Compile into a one-page decision memo
- [ ] Internal review: does the memo excite a non-expert?

### Go/No-Go Gate

**Proceed if:** The one-sentence claim is sharp, the diagnostics are measurable, and the classical kill is well-defined.  
**Kill if:** Cannot write a coherent sentence distinguishing this from classical thermalization after one week of iteration.

---

## Phase 2: Build Proof Architecture

**Duration estimate:** 1-2 weeks  
**Dependencies:** Phase 1 complete

### Checklist

#### Hamiltonian Family
- [ ] Choose the coupling topology (e.g., random subgraph of Zephyr/Pegasus, or structured lattice)
- [ ] Define the disorder distribution (uniform on [-W, W], Gaussian, or binary)
- [ ] Define the S-E partition strategy (geometric, random, or connectivity-based)
- [ ] Verify that the chosen Hamiltonian family is embeddable on both Advantage2 and Advantage_system6.4

#### Observables
- [ ] Code the initial-state distance metric: D(P_S|init_1, P_S|init_2)
- [ ] Code the Gibbs-fit procedure: P_S vs exp(-beta_eff H_S^eff)/Z
- [ ] Code the memory order parameter
- [ ] Implement bootstrap error bars for all observables
- [ ] Validate on synthetic data (known distributions)

#### Classical Baselines
- [ ] Exact diagonalization code (sparse Hamiltonian, Lanczos or full diag)
  - [ ] Validate on 2-4 qubit systems with known spectra
  - [ ] Benchmark memory usage for 16, 18, 20 qubits on M3 Max
- [ ] Lindblad master equation solver
  - [ ] Choose jump operators (single-qubit relaxation and dephasing)
  - [ ] Validate on known open-system benchmarks
- [ ] Classical spin-vector Monte Carlo
  - [ ] Implement Metropolis-Hastings for classical Ising model
  - [ ] Validate against exact solutions (1D Ising, small 2D Ising)
- [ ] Glauber dynamics
  - [ ] Implement single-spin-flip dynamics at temperature T
  - [ ] Validate thermalization time against analytical predictions (1D Ising)

#### D-Wave Interface
- [ ] Write reverse-anneal job submission wrapper
- [ ] Implement metadata logging (anneal schedule, initial state, QPU ID, calibration data, timing, disorder realization)
- [ ] Test on Advantage2 with a trivial problem (2-qubit ferromagnet)
- [ ] Test on Advantage_system6.4 with the same trivial problem
- [ ] Verify that `reinitialize_state=True` works as expected
- [ ] Measure and log QPU timing overhead

#### Metadata Schema
- [ ] Define JSON/HDF5 schema for raw data
- [ ] Fields: job_id, qpu_id, anneal_schedule, initial_state, h, J, disorder_realization, num_reads, timing_info, calibration_epoch, raw_samples
- [ ] Write data ingestion and validation scripts

### Go/No-Go Gate

**Proceed if:** All code validates on toy problems. Metadata schema is finalized. Both QPUs are accessible and responding correctly.  
**Kill if:** Critical code bugs persist after validation. QPU access is lost for extended period.

---

## Phase 3: Run Easy Discovery Version

**Duration estimate:** 2-3 weeks  
**Dependencies:** Phase 2 complete

### Checklist

#### Initial State Preparation
- [ ] Define initial state set: {all-up, all-down, random_1, random_2, domain_wall, Neel}
- [ ] Verify that all initial states are valid on both QPUs
- [ ] Implement initial state rotation for the subsystem-only case (Phase 4 prep)

#### Parameter Scans
- [ ] **Scan 1:** t_p sweep at fixed s_p = 0.4, N = 50, W = 0
  - [ ] t_p = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] μs
  - [ ] 2+ initial states, 1000+ shots each
- [ ] **Scan 2:** s_p sweep at fixed t_p = 100 μs, N = 50, W = 0
  - [ ] s_p = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
  - [ ] 2+ initial states, 1000+ shots each
- [ ] **Scan 3:** N sweep at fixed s_p = 0.4, t_p = 100 μs, W = 0
  - [ ] N = [8, 16, 32, 50, 75, 100, 150, 200]
  - [ ] 2+ initial states, 1000+ shots each
- [ ] **Scan 4:** W sweep at fixed s_p = 0.4, t_p = 100 μs, N = 50
  - [ ] W = [0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
  - [ ] 2+ initial states, 1000+ shots each
- [ ] **Scan 5:** Return-ramp shape comparison
  - [ ] Fast return (1 μs), medium (5 μs), slow (20 μs)
  - [ ] Fixed s_p = 0.4, t_p = 100 μs, N = 50, W = 0

#### Analysis
- [ ] Compute initial-state distance for each scan point
- [ ] Compute Gibbs-fit quality for each scan point
- [ ] Plot all scans with error bars
- [ ] Check reproducibility: repeat selected scan points on different days
- [ ] Identify the "interesting regime" where thermalization is clearly occurring

### Go/No-Go Gate

**Proceed if:** Initial-state dependence clearly decreases with increasing t_p and/or decreasing s_p. Gibbs fits are reasonable in the thermalized regime. Disorder arrests the process.  
**Kill if:** No initial-state dependence decrease at any parameter point. Kill the project entirely.  
**Pivot if:** Gibbs fits are bad but initial-state distance decreases — the physics may be different from Gibbs thermalization. Investigate before proceeding.

---

## Phase 4: Upgrade to Subsystem-Bath Experiment

**Duration estimate:** 3-4 weeks  
**Dependencies:** Phase 3 shows positive results

### Checklist

#### Partition Design
- [ ] Choose subsystem S: connected subgraph of |S| = 4-8 qubits
- [ ] Define environment E: remaining qubits coupled to S
- [ ] Implement lambda-controlled S-E coupling (scale couplers crossing the S-E boundary by lambda)
- [ ] Verify partition is embeddable on both QPUs

#### Core Scans
- [ ] **Scan A:** |E| sweep at fixed |S|, lambda, s_p, t_p, W = 0
  - [ ] |E| = [4, 8, 16, 32, 50, 75, 100]
  - [ ] 3+ initial states of S, 2000+ shots each
  - [ ] Compute P_S for each initial state and |E|
- [ ] **Scan B:** lambda sweep at fixed |S|, |E|, s_p, t_p, W = 0
  - [ ] lambda = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
  - [ ] 3+ initial states of S, 2000+ shots each
- [ ] **Scan C:** W sweep at fixed |S|, |E|, lambda, s_p, t_p
  - [ ] W = [0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
  - [ ] 3+ initial states of S, 2000+ shots each
- [ ] **Scan D:** 2D grid (|E| or lambda) vs W for crossover diagram
  - [ ] Grid resolution: ~10x10 points minimum
  - [ ] 2+ initial states, 1000+ shots per point

#### Analysis
- [ ] For each scan point: compute P_S marginal over S for each initial state
- [ ] Compute initial-state distance on P_S (not on full P)
- [ ] Compute Gibbs-fit quality for P_S
- [ ] Extract effective temperature beta_eff for P_S
- [ ] Check that beta_eff is consistent across initial states (key diagnostic)
- [ ] Plot convergence of P_S across initial states as |E| grows
- [ ] Construct crossover diagram in (|E| or lambda, W) plane

### Go/No-Go Gate

**Proceed if:** P_S clearly converges across initial states with growing |E| and lambda. beta_eff is consistent. Disorder arrests convergence.  
**Downgrade to Nature Physics if:** Convergence is visible but noisy, or beta_eff is not perfectly consistent.  
**Kill if:** P_S never converges. Subsystem thermalization is not occurring.

---

## Phase 5: Kill Classical Explanations

**Duration estimate:** 2-3 weeks  
**Dependencies:** Phase 4 shows positive results

### Checklist

#### Exact Diagonalization
- [ ] Run ED for the same Hamiltonian at small sizes (|S|+|E| ≤ 20)
- [ ] Compute P_S from time-evolved pure states
- [ ] Compare ED P_S with annealer P_S — should agree if the annealer is doing quantum thermalization
- [ ] If agreement: strong support for quantum interpretation
- [ ] If disagreement: investigate (could be decoherence effects, anneal schedule artifacts, or a bug)

#### Lindblad Master Equation
- [ ] Run Lindblad for small systems (|S|+|E| ≤ 14)
- [ ] Try several bath models (weak coupling, strong coupling, different spectral densities)
- [ ] Compare with annealer data
- [ ] Key question: can a Markovian bath reproduce the |E| and lambda dependence?

#### Classical Spin-Vector Monte Carlo
- [ ] Run classical MC at T = T_device for the same Hamiltonian
- [ ] Compute same observables (initial-state distance, Gibbs fit)
- [ ] Compare thermalization timescale and crossover location with annealer
- [ ] Key question: does the annealer thermalize faster or at different parameter values than classical MC?

#### Glauber Dynamics
- [ ] Run Glauber at T = T_device
- [ ] Match the timescale to the annealer pause time (map Glauber steps to microseconds)
- [ ] Compare all three diagnostics
- [ ] **This is the critical comparison.** If Glauber reproduces everything, the paper is dead for Nature.

#### Hardware Control Sweeps
- [ ] Vary anneal_time (ramp speed) and check if thermalization changes
- [ ] Compare Advantage2 vs Advantage_system6.4 for the same Hamiltonian parameters
- [ ] Check whether the crossover location shifts consistently with the energy scale ratio

### Go/No-Go Gate

**Proceed to Phase 6 if:** At least one classical baseline clearly fails. ED agrees with annealer (supporting quantum interpretation). The failure is robust across parameter space.  
**Downgrade if:** Classical baselines come close but not exact — ambiguous quantum advantage.  
**Kill for Nature if:** Glauber at T_device reproduces all features.

---

## Phase 6: Universal Physics Framing

**Duration estimate:** 1-2 weeks  
**Dependencies:** Phase 5 results available

### Checklist

- [ ] Identify the best order parameter (e.g., initial-state distance D, Gibbs residual G, or a combination)
- [ ] Identify the best control parameter (|E|, lambda, |E|*lambda, or a rescaled combination)
- [ ] Plot order parameter vs control parameter at several disorder strengths
- [ ] Identify the crossover line in the (control, W) plane
- [ ] Fit the crossover to a functional form if possible (power law, exponential, scaling collapse)
- [ ] Connect to theoretical predictions:
  - [ ] ETH: subsystem thermalization should hold when |E| >> |S| and coupling is not too weak
  - [ ] MBL: disorder should arrest thermalization above a critical W_c
  - [ ] Any finite-size scaling or system-size dependence
- [ ] Verify universality: same crossover (in dimensionless units) on both QPUs
- [ ] Write the physics narrative: "We observe a crossover from thermalized to localized behavior, controlled by [parameter], consistent with [ETH/MBL predictions], on a programmable quantum platform."

### Go/No-Go Gate

**Proceed if:** Clean crossover with identifiable order and control parameters. Consistent across QPUs in dimensionless units.  
**Downgrade if:** Crossover is messy, device-dependent, or does not connect to known theory.

---

## Phase 7: Write for Editors

**Duration estimate:** 2-4 weeks  
**Dependencies:** Phase 6 complete

### Checklist

#### Summary (First)
- [ ] Write 150-200 word summary answering all five questions
- [ ] Get feedback from 2+ colleagues
- [ ] Iterate until a non-expert physicist says "interesting"

#### Figures (Second)
- [ ] Finalize Figure 1: Protocol and setup
- [ ] Finalize Figure 2: Discovery-phase data
- [ ] Finalize Figure 3: Subsystem thermalization
- [ ] Finalize Figure 4: Disorder arrest + crossover
- [ ] Finalize Figure 5: Classical baseline comparison
- [ ] Create figure storyboard: read figures 1-5 in sequence — does it tell the story without text?

#### Main Text (Third)
- [ ] Introduction: big question, what's missing, what's new (~400 words)
- [ ] Results: walk through figures 1-5 (~1500 words)
- [ ] Discussion: why it matters, connection to ETH/MBL, outlook (~500 words)
- [ ] Methods: experimental protocol, Hamiltonian, observables, classical baselines (~1000 words)

#### Supplementary
- [ ] Extended methods
- [ ] Additional parameter scans
- [ ] All classical baseline details and comparisons
- [ ] Raw data availability statement

### Go/No-Go Gate

**Proceed if:** Summary is compelling. Figures tell the story. Text is tight.  
**Iterate if:** Summary is not exciting. Go back to framing.

---

## Phase 8: Submission Strategy

**Duration estimate:** 1-2 weeks (after writing)  
**Dependencies:** Phase 7 complete

### Checklist

- [ ] Evaluate the 5 Nature criteria:
  - [ ] Criterion 1 (subsystem thermalization): Met / Not met
  - [ ] Criterion 2 (disorder arrest): Met / Not met
  - [ ] Criterion 3 (classical kill): Met / Not met
  - [ ] Criterion 4 (universal framing): Met / Not met
  - [ ] Criterion 5 (cross-QPU validation): Met / Not met
- [ ] Decision:
  - [ ] All 5 → submit to Nature
  - [ ] 1-2 met, 3-5 partial → submit to Nature Physics
  - [ ] 1-2 met, 3-5 weak → submit to Nature Communications
  - [ ] Only 1 → submit to Communications Physics or PRL
- [ ] Prepare cover letter tailored to target journal
- [ ] Identify suggested reviewers (experts in quantum thermalization, MBL, quantum annealing)
- [ ] Identify excluded reviewers (if any conflicts)
- [ ] Format manuscript to journal requirements
- [ ] Submit
- [ ] Prepare rapid-response rebuttal template for common referee objections

### Common Referee Objections to Anticipate

1. "This could be classical thermal noise" → Point to Phase 5 results
2. "The system is open, not isolated" → Acknowledge, but show self-thermalization dominates over external bath
3. "The Hilbert space is too large for proper ED comparison" → Acknowledge limitation, emphasize scaling consistency
4. "This is engineering, not physics" → Point to universal framing and ETH/MBL connection
5. "Only two QPU platforms is not enough for universality" → Acknowledge, discuss in outlook, emphasize dimensionless scaling

