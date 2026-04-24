# D-Wave Hardware Reference

Hardware specifications and reverse-anneal API reference for the locth1 project.

---

## QPU Comparison

| Property | Advantage2 (Primary) | Advantage_system6.4 (Secondary) |
|----------|----------------------|----------------------------------|
| **Solver name** | Advantage2_prototype2.6 | Advantage_system6.4 |
| **Topology** | Zephyr | Pegasus |
| **Working qubits** | 4,579 | 5,612 |
| **Working couplers** | [TO BE DETERMINED — query solver properties] | [TO BE DETERMINED] |
| **Energy scale (h_max)** | 2.308 GHz | 1.281 GHz |
| **Effective temperature (kT/h)** | 0.112 | 0.221 |
| **J/kT ratio** | ~20.6 | ~5.8 |
| **Qubit connectivity** | ~20 (Zephyr degree) | ~15 (Pegasus degree) |
| **Anneal time range** | [TO BE DETERMINED] | [TO BE DETERMINED] |
| **Max anneal schedule points** | [TO BE DETERMINED — check solver.properties] | [TO BE DETERMINED] |
| **Reverse annealing support** | Yes | Yes |
| **Role in this paper** | Primary data (higher quantum character) | Cross-validation (universality test) |

### Why These Two QPUs

The two QPUs have different energy scales and effective temperatures, which creates a natural test of universality:

1. **Advantage2** has a ~1.8x higher energy scale and ~2x lower effective temperature. The J/kT ratio (~20.6) means it operates deeper in the quantum regime for the same anneal schedule parameter s. This makes it the preferred platform for detecting quantum thermalization signatures.

2. **Advantage_system6.4** has more qubits (5,612 vs 4,579) but weaker couplings and higher temperature. If thermalization phenomena appear at the same dimensionless parameter values (W/J, lambda*J/kT, |E|) on both platforms, the physics is universal and not an artifact of one specific QPU.

### Topology Considerations

**Zephyr (Advantage2):**
- Higher connectivity (~20 neighbors per qubit)
- More compact embeddings possible
- Newer topology, less well-characterized in the literature

**Pegasus (Advantage_system6.4):**
- Moderate connectivity (~15 neighbors per qubit)
- Well-characterized topology
- More qubits available

**For this experiment:** We need a connected subgraph that is embeddable on BOTH topologies. Options:
- Use a common subgraph (e.g., a chimera-like structure that embeds natively on both)
- Use minor embedding (software layer) to embed the same logical graph on both
- Use a simple structure (e.g., complete graph K_n for small n, or a random regular graph) with minor embedding on both

---

## Reverse Annealing API Reference

### Core Parameters

#### `initial_state`
A dictionary mapping qubit indices (physical or logical) to spin values.

```python
# Physical qubit example
initial_state = {0: 1, 1: -1, 2: 1, 3: -1, 4: 1}

# With EmbeddingComposite, use logical qubit labels
initial_state = {'q0': 1, 'q1': -1, 'q2': 1}
```

**Requirements:**
- Must include every qubit used in the problem
- Values must be +1 or -1
- For this experiment, we need multiple initial states to compare (e.g., all-up, all-down, random, domain wall)

#### `anneal_schedule`
A list of (time_in_microseconds, s_value) pairs defining the piecewise-linear anneal trajectory.

```python
# Standard reverse-anneal schedule
anneal_schedule = [
    (0.0, 1.0),                          # start at s=1 (classical Ising)
    (t_ramp_down, s_p),                   # ramp down to pause point
    (t_ramp_down + t_hold, s_p),          # hold at pause point
    (t_ramp_down + t_hold + t_ramp_up, 1.0)  # ramp back to s=1
]
```

**Requirements:**
- Times must be non-negative and strictly increasing
- s values must be between 0.0 and 1.0
- For reverse annealing, must start at s=1.0 and end at s=1.0
- Number of points must not exceed `solver.properties['max_anneal_schedule_points']`
- Cannot be used simultaneously with `annealing_time`

#### `reinitialize_state`
Boolean flag controlling whether the system is reinitialized before each shot.

| Value | Behavior | Use Case |
|-------|----------|----------|
| `True` | Each shot starts from `initial_state` | **This experiment** — measuring initial-state dependence |
| `False` | Each shot starts from the result of the previous shot | Iterated reverse annealing (not used here) |

**Critical for this experiment:** Always set to `True`. We need each shot to start from the specified initial state so that we can measure whether the output distribution depends on the initial state.

### Timing Controls

| Parameter | Type | Description |
|-----------|------|-------------|
| `annealing_time` | float (μs) | Total anneal time for a standard forward anneal. **Do not use with `anneal_schedule`.** |
| `readout_thermalization` | float (μs) | Delay between anneal completion and readout. Reduces readout errors. |
| `reduce_intersample_correlation` | bool | If True, adds delay between samples to reduce correlations. |
| `programming_thermalization` | float (μs) | Delay after programming h/J values before starting the anneal. |
| `num_reads` | int | Number of samples per job submission. Max varies by solver. |

### Sample Code: Full Reverse-Anneal Workflow

```python
import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
import json
import datetime

# ─── Configuration ───────────────────────────────────────────────
QPU_SOLVER = 'Advantage2_prototype2.6'  # or 'Advantage_system6.4'
N = 50           # total qubits
S_SIZE = 6       # subsystem size
s_p = 0.4        # pause point
t_ramp = 5.0     # ramp time (μs)
t_hold = 100.0   # hold time (μs)
W = 0.0          # disorder strength
LAMBDA = 0.5     # S-E coupling
NUM_READS = 2000

# ─── Build Hamiltonian ──────────────────────────────────────────
# Define qubits and couplings for S and E
# (Details depend on topology and partition strategy)
S_qubits = list(range(S_SIZE))
E_qubits = list(range(S_SIZE, N))

# Random fields (disorder)
rng = np.random.default_rng(seed=42)
h = {i: rng.uniform(-W, W) for i in range(N)}

# Couplings within S, within E, and between S-E
J = {}
# ... (populate based on connectivity graph)
# Scale S-E couplers by lambda
for (i, j) in SE_edges:
    J[(i, j)] *= LAMBDA

# ─── Define Initial States ──────────────────────────────────────
initial_states = {
    'all_up':    {i: +1 for i in range(N)},
    'all_down':  {i: -1 for i in range(N)},
    'random_1':  {i: int(rng.choice([-1, 1])) for i in range(N)},
    'neel':      {i: (-1)**i for i in range(N)},
}

# ─── Define Anneal Schedule ─────────────────────────────────────
anneal_schedule = [
    (0.0, 1.0),
    (t_ramp, s_p),
    (t_ramp + t_hold, s_p),
    (2 * t_ramp + t_hold, 1.0),
]

# ─── Run Reverse Anneals ────────────────────────────────────────
sampler = DWaveSampler(solver=QPU_SOLVER)
# Optionally wrap with EmbeddingComposite for logical problems

results = {}
for name, init_state in initial_states.items():
    response = sampler.sample_ising(
        h, J,
        initial_state=init_state,
        anneal_schedule=anneal_schedule,
        reinitialize_state=True,
        num_reads=NUM_READS,
        label=f'locth1_{name}_sp{s_p}_tp{t_hold}_W{W}',
    )

    # Store results
    results[name] = {
        'samples': response.record.sample.tolist(),
        'energies': response.record.energy.tolist(),
        'timing': response.info.get('timing', {}),
    }

# ─── Compute P_S ────────────────────────────────────────────────
from collections import Counter

def compute_P_S(samples, S_indices):
    """Compute marginal distribution over subsystem S."""
    S_configs = [tuple(s[i] for i in S_indices) for s in samples]
    counts = Counter(S_configs)
    total = sum(counts.values())
    return {config: count / total for config, count in counts.items()}

P_S = {}
for name, data in results.items():
    P_S[name] = compute_P_S(data['samples'], S_qubits)

# ─── Compute Initial-State Distance ─────────────────────────────
def total_variation_distance(P, Q):
    """Total variation distance between two distributions."""
    all_keys = set(P.keys()) | set(Q.keys())
    return 0.5 * sum(abs(P.get(k, 0) - Q.get(k, 0)) for k in all_keys)

D_up_down = total_variation_distance(P_S['all_up'], P_S['all_down'])
print(f"TVD(all_up, all_down) on S: {D_up_down:.4f}")

# ─── Save Metadata ──────────────────────────────────────────────
metadata = {
    'timestamp': datetime.datetime.now().isoformat(),
    'solver': QPU_SOLVER,
    'N': N, 'S_SIZE': S_SIZE,
    's_p': s_p, 't_hold': t_hold, 't_ramp': t_ramp,
    'W': W, 'lambda': LAMBDA,
    'num_reads': NUM_READS,
    'anneal_schedule': anneal_schedule,
    'initial_states': list(initial_states.keys()),
    'D_up_down': D_up_down,
}
with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

### Solver Property Queries

Before running experiments, query the solver for hardware-specific constraints:

```python
sampler = DWaveSampler(solver=QPU_SOLVER)
props = sampler.properties

# Key properties to check
print(f"Qubits: {len(sampler.nodelist)}")
print(f"Couplers: {len(sampler.edgelist)}")
print(f"Max anneal schedule points: {props.get('max_anneal_schedule_points')}")
print(f"Annealing time range: {props.get('annealing_time_range')}")
print(f"Default annealing time: {props.get('default_annealing_time')}")
print(f"h range: {props.get('h_range')}")
print(f"J range: {props.get('j_range')}")
print(f"Max num reads: {props.get('num_reads_range')}")
print(f"Extended J range: {props.get('extended_j_range')}")

# Anneal schedule details
print(f"Anneal offset ranges: {props.get('anneal_offset_ranges')}")

# Check reverse annealing support
print(f"Supports reverse annealing: {'initial_state' in sampler.parameters}")
```

### Important API Notes

1. **`anneal_schedule` vs `annealing_time`**: These are mutually exclusive. When using reverse annealing, always use `anneal_schedule` and do not set `annealing_time`.

2. **Schedule constraints**: The schedule must be piecewise linear with strictly increasing times. The s values must be monotonically changing within each segment (but can change direction at breakpoints).

3. **`initial_state` completeness**: Every qubit in the problem must appear in `initial_state`. Missing qubits will cause an error.

4. **Embedding and `initial_state`**: When using `EmbeddingComposite`, the `initial_state` should use logical qubit labels. The embedding composite handles the mapping to physical qubits internally.

5. **`label` parameter**: Use descriptive labels for job tracking. The D-Wave dashboard shows these labels, making it easy to find and organize jobs.

6. **Rate limits**: D-Wave has per-minute and per-hour rate limits. For large scan campaigns, implement throttling and error handling.

7. **Calibration epochs**: The QPU is recalibrated periodically. Log the calibration epoch (available in `response.info`) to check for calibration-dependent effects.

---

## Classical Hardware

| Component | Spec | Use |
|-----------|------|-----|
| Chip | Apple M3 Max | All classical computation |
| RAM | 128 GB unified | Exact diag up to ~20 qubits, Lindblad up to ~14 qubits |
| GPU cores | 40 | MPS backend for PyTorch-based calculations |
| Framework | PyTorch (MPS backend) | Lindblad solver, data analysis |
| Additional | NumPy, SciPy (sparse) | Exact diag, Monte Carlo |

### Memory Estimates for Exact Diagonalization

| Qubits | Hilbert dim | State vector (complex128) | Hamiltonian (sparse, ~10 nnz/row) | Feasible on M3 Max? |
|--------|------------|--------------------------|-----------------------------------|-----------------------|
| 16 | 65,536 | 1 MB | ~10 MB | Yes |
| 18 | 262,144 | 4 MB | ~40 MB | Yes |
| 20 | 1,048,576 | 16 MB | ~160 MB | Yes |
| 22 | 4,194,304 | 64 MB | ~640 MB | Yes |
| 24 | 16,777,216 | 256 MB | ~2.5 GB | Marginal |
| 26 | 67,108,864 | 1 GB | ~10 GB | Tight but possible |
| 28 | 268,435,456 | 4 GB | ~40 GB | No (for full diag) |

For Lindblad (density matrix = dim^2):

| Qubits | Density matrix dim | Memory (complex128) | Feasible? |
|--------|-------------------|---------------------|-----------|
| 10 | 1,024 x 1,024 | 16 MB | Yes |
| 12 | 4,096 x 4,096 | 256 MB | Yes |
| 14 | 16,384 x 16,384 | 4 GB | Yes |
| 16 | 65,536 x 65,536 | 64 GB | Marginal |
| 18 | 262,144 x 262,144 | 1 TB | No |

