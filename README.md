# locth1 — Quantum Annealer Self-Thermalization

**Target journal:** Nature (primary) / Nature Physics (fallback)

## Flagship Claim

> Above a tunable size/coupling threshold, a programmable quantum annealer acts as its own bath and drives subsystem thermalization; quenched disorder arrests this process and preserves memory.

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Define the Nature sentence | Not started |
| 2 | Build proof architecture | Not started |
| 3 | Run easy discovery version | Not started |
| 4 | Upgrade to subsystem-bath experiment | Not started |
| 5 | Kill classical explanations | Not started |
| 6 | Universal physics framing | Not started |
| 7 | Write for editors | Not started |
| 8 | Submission strategy | Not started |

## Project Files

| File | Contents |
|------|----------|
| [analysis.md](analysis.md) | Full 10-section structured analysis of the paper plan |
| [phases.md](phases.md) | Detailed phase-by-phase execution plan with checklists |
| [parameters.md](parameters.md) | Reference table of all experimental parameters |
| [hardware.md](hardware.md) | D-Wave hardware comparison and reverse-anneal API reference |

## Key Hardware

- **Primary QPU:** Advantage2 (4579 qubits, 2.308 GHz energy scale, T_eff = 0.112)
- **Secondary QPU:** Advantage_system6.4 (5612 qubits, 1.281 GHz energy scale, T_eff = 0.221)
- **Classical compute:** M3 Max 128 GB (MPS backend for PyTorch)

## Core Experimental Technique

Reverse annealing with controlled pause: prepare initial state, anneal backward to pause point s_p, hold for time t_p, then return. Measure whether subsystem S thermalizes as environment E grows.

## Submission Ladder

Nature → Nature Physics → Nature Communications → Communications Physics / PRL
