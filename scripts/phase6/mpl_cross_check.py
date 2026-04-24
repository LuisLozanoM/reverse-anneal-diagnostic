"""Maximum-pseudolikelihood cross-check of the single-qubit-probe β_eff.

For every condition in the Phase 3C disorder sweep (60 conditions per QPU,
10 seeds × 6 disorder strengths), load the saved QPU samples from the h5
artefacts and estimate β via the maximum-pseudolikelihood temperature
estimator on the programmed (h, J).  Compare the distribution of per-
condition β_MPL against the single-qubit-probe calibration (β_probe =
7.219 for Advantage2, 4.289 for Advantage_system6.4).

This is the cross-check that the quantum-annealer-playbook skill asks
for: two independent β estimators that should agree to within ~10%.
"""

from __future__ import annotations

import ast
import json
import statistics
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from locth1.qpu_utils import estimate_mpl_effective_temperature  # noqa: E402


def _parse_key(k):
    return ast.literal_eval(k) if isinstance(k, str) else k


def _coerce_h(h_raw):
    return {int(_parse_key(k)): float(v) for k, v in h_raw.items()}


def _coerce_j(J_raw):
    out = {}
    for k, v in J_raw.items():
        if isinstance(k, str):
            pair = ast.literal_eval(k)
        else:
            pair = k
        out[(int(pair[0]), int(pair[1]))] = float(v)
    return out


def _load_samples(raw_path: Path, variables: list[int]):
    with h5py.File(raw_path, "r") as hf:
        samples = np.asarray(hf["samples"], dtype=np.int8)
    return samples


def run_cross_check(
    summary_path: Path,
    max_conditions: int | None = None,
    relaxed_only: bool = False,
    relaxed_threshold: float = 0.05,
    min_W: float = 0.0,
    min_entropy: float = 0.0,
):
    summary = json.loads(summary_path.read_text())
    beta_probe = float(summary["beta_eff"])
    solver = summary.get("solver", "unknown")
    conditions = summary.get("conditions", [])
    if max_conditions is not None:
        conditions = conditions[:max_conditions]

    rows = []
    for cond in conditions:
        if cond.get("W", 0.0) < min_W:
            continue
        h = _coerce_h(cond["h"])
        J = _coerce_j(cond["J"])
        variables = sorted(h.keys())
        per_init = cond.get("per_initial_state", [])

        # Memory order parameter from pooled P_S across initial states
        ps = [p.get("P_S", {}) for p in per_init]
        all_keys = set().union(*(p.keys() for p in ps))
        def dist(p):
            return np.array([p.get(k, 0.0) for k in all_keys])
        memory = 0.0
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                memory = max(memory, 0.5 * np.abs(dist(ps[i]) - dist(ps[j])).sum())

        if relaxed_only and memory > relaxed_threshold:
            continue

        # Filter on pooled entropy so that MPL has a non-degenerate spectrum
        # to fit.  W=0 ferromagnetic conditions concentrate on a single ordered
        # state, giving β_MPL = ∞ trivially.
        pooled_dist = sum([dist(p) for p in ps], np.zeros(len(all_keys)))
        pooled_dist = pooled_dist / pooled_dist.sum()
        nz = pooled_dist[pooled_dist > 0]
        entropy = float(-np.sum(nz * np.log(nz)))
        if entropy < min_entropy:
            continue

        all_samples = []
        for entry in per_init:
            raw = REPO / entry["raw_path"]
            if not raw.exists():
                continue
            s = _load_samples(raw, variables)
            all_samples.append(s)
        if not all_samples:
            continue
        pooled = np.vstack(all_samples)

        est = estimate_mpl_effective_temperature(h, J, pooled, variables=variables)
        beta_mpl = float(est.beta)
        rows.append({
            "W": cond["W"],
            "seed": cond["seed"],
            "N": cond["N"],
            "memory": float(memory),
            "entropy_nats": float(entropy),
            "n_samples": int(pooled.shape[0]),
            "beta_mpl": beta_mpl,
            "beta_probe": beta_probe,
            "ratio_mpl_over_probe": beta_mpl / beta_probe if np.isfinite(beta_mpl) else None,
        })
    return {"solver": solver, "beta_probe": beta_probe, "rows": rows}


def summarise(result):
    rows = result["rows"]
    if not rows:
        print("No conditions matched.")
        return
    betas = [r["beta_mpl"] for r in rows if np.isfinite(r["beta_mpl"])]
    ratios = [r["ratio_mpl_over_probe"] for r in rows if r["ratio_mpl_over_probe"] is not None]
    n_inf = sum(1 for r in rows if not np.isfinite(r["beta_mpl"]))
    print(f"Solver: {result['solver']}")
    print(f"β_probe (single-qubit): {result['beta_probe']:.4f}")
    print(f"Conditions evaluated: {len(rows)}  (finite β_MPL: {len(betas)}, degenerate: {n_inf})")
    if betas:
        print(f"β_MPL  median: {statistics.median(betas):.4f}")
        print(f"β_MPL  mean:   {statistics.mean(betas):.4f}")
        if len(betas) > 1:
            print(f"β_MPL  stdev:  {statistics.pstdev(betas):.4f}")
        print(f"β_MPL / β_probe  median: {statistics.median(ratios):.3f}")
        print(f"β_MPL / β_probe  range:  [{min(ratios):.3f}, {max(ratios):.3f}]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="data/raw/phase6_gibbs/phase3c_disorder_sweep/summary.json")
    parser.add_argument("--max-conditions", type=int, default=None)
    parser.add_argument("--relaxed-only", action="store_true")
    parser.add_argument("--relaxed-threshold", type=float, default=0.05)
    parser.add_argument("--min-W", type=float, default=0.0)
    parser.add_argument("--min-entropy", type=float, default=0.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = run_cross_check(
        REPO / args.summary,
        max_conditions=args.max_conditions,
        relaxed_only=args.relaxed_only,
        relaxed_threshold=args.relaxed_threshold,
        min_W=args.min_W,
        min_entropy=args.min_entropy,
    )
    summarise(result)
    if args.out:
        out_path = REPO / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Wrote {out_path}")
