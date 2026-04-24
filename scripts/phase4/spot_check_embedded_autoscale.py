"""Embedded auto_scale spot-check with a manually-bounded chain strength.

The legacy Phase 3/4 main sweeps used `EmbeddingComposite` with the
SDK-default `uniform_torque_compensation` chain strength, which can push
chain couplers above the solver's |J|=1 range; auto_scale=True then
rescaled every (h, J, chain coupler) to fit the range before programming.
This script submits a logical problem twice through a *fixed* embedding
with `chain_strength=1.0` (so chains just fit the j_range):
once with auto_scale=True, once with auto_scale=False.  Any residual
memory-order-parameter difference isolates the effect of auto_scale's
rescaling on the logical (h, J).

The output archives sampleset.info for each run so the chain-break
fraction and the actual programmed problem can be inspected.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from locth1.analysis import compute_marginal  # noqa: E402
from locth1.observables import memory_order_parameter  # noqa: E402


SUBSYSTEM_SIZE = 6
S_P = 0.4
T_HOLD = 100.0
NUM_READS = 2000
DISORDER_SEED = 42
CHAIN_STRENGTH = 2.0  # uses extended_j_range=[-2,1]; closer to uniform_torque_compensation defaults


SPOT_CONDITIONS = [
    # 4-tuples interpreted as ferromagnetic, |S|=6, chain_strength=CHAIN_STRENGTH.
    # See also SPOT_CONDITIONS_EX for Phase 7 reverify + UTC disorder sweep.
]

# Extended conditions: (label, n_env, lam, W, mode, subsystem_size,
# sign_seed_or_None, chain_strength_override_or_None)
#
# mode: "ferro" or "frustrated"
# sign_seed: if frustrated, the random seed used to flip subsystem-internal
#   coupler signs with probability 0.5 (|S|-only Edwards-Anderson frustration)
# chain_strength_override: None -> use EmbeddingComposite default (uniform_torque_compensation);
#   a float overrides with that value.
SPOT_CONDITIONS_EX = [
    # UTC disorder-arrest on Advantage2 — multiple seeds per W for averaging.
    ("fig3a_utc_W0.0_s0", 50, 0.5, 0.0, "ferro", 6, None, None),
    ("fig3a_utc_W0.5_s0", 50, 0.5, 0.5, "ferro", 6, None, None),
    ("fig3a_utc_W1.0_s0", 50, 0.5, 1.0, "ferro", 6, None, None),
    ("fig3a_utc_W1.5_s0", 50, 0.5, 1.5, "ferro", 6, None, None),
    ("fig3a_utc_W2.0_s0", 50, 0.5, 2.0, "ferro", 6, None, None),
]


def _build_logical(n_env, lam, W, seed=DISORDER_SEED, subsystem_size=SUBSYSTEM_SIZE,
                   sign_seed=None):
    """Logical random-3-regular problem.

    If sign_seed is None, subsystem-internal couplers are ferromagnetic.
    If sign_seed is an int, subsystem-internal couplers have their signs
    flipped independently with probability 0.5 (|S|-Edwards-Anderson
    frustration).  Environment couplers remain ferromagnetic.
    """
    n_total = subsystem_size + n_env
    G = nx.random_regular_graph(3, n_total, seed=seed)
    visited = [0]
    seen = {0}
    queue = [0]
    while queue and len(visited) < subsystem_size:
        node = queue.pop(0)
        for nbr in sorted(G.neighbors(node)):
            if nbr not in seen:
                seen.add(nbr)
                visited.append(nbr)
                queue.append(nbr)
                if len(visited) >= subsystem_size:
                    break
    S = visited[:subsystem_size]
    S_set = set(S)
    E = [n for n in G.nodes() if n not in S_set]
    rng = np.random.default_rng(seed)
    sign_rng = np.random.default_rng(sign_seed) if sign_seed is not None else None
    h = {n: float(rng.uniform(-W, W)) for n in G.nodes()}
    J = {}
    for u, v in G.edges():
        boundary = (u in S_set) != (v in S_set)
        if boundary:
            Jij = -1.0 * lam
        elif u in S_set and v in S_set:
            if sign_rng is not None:
                # Edwards-Anderson bond disorder inside S
                sign = 1.0 if sign_rng.random() < 0.5 else -1.0
                Jij = sign * 1.0
            else:
                Jij = -1.0
        else:
            Jij = -1.0
        J[(u, v)] = Jij
    return h, J, S, E


def _initial_states(S, E, seed=DISORDER_SEED):
    rng = np.random.default_rng(seed)
    E_state = {q: 1 for q in E}
    return {
        "S_up":     {**{q: 1 for q in S},  **E_state},
        "S_down":   {**{q: -1 for q in S}, **E_state},
        "S_random": {**{q: int(rng.choice([-1, 1])) for q in S}, **E_state},
    }


def _run_arm_with_embedding_composite(sampler, h, J, S, E, auto_scale, chain_strength):
    """Run one arm using EmbeddingComposite with default embedding + chain strength.

    Used when chain_strength=None to pick up uniform_torque_compensation default.
    """
    from dwave.system import EmbeddingComposite
    composite = EmbeddingComposite(sampler)
    P_S = {}
    chain_breaks = []
    for name, init in _initial_states(S, E).items():
        sample_kwargs = dict(
            num_reads=NUM_READS,
            anneal_schedule=_schedule(),
            initial_state=dict(init),
            reinitialize_state=True,
            auto_scale=auto_scale,
            answer_mode="raw",
            label=f"utc_as_{'T' if auto_scale else 'F'}_{name}",
        )
        if chain_strength is not None:
            sample_kwargs["chain_strength"] = chain_strength
        ss = composite.sample_ising(h, J, **sample_kwargs)
        if hasattr(ss, "resolve"):
            ss.resolve()
        variables = list(ss.variables)
        samples = np.asarray(ss.record.sample, dtype=np.int8)
        P_S[name] = compute_marginal(samples, S, variables=variables)
        info = dict(ss.info or {})
        cb = info.get("embedding_context", {}).get("chain_break_fraction")
        if cb is None and hasattr(ss.record, "chain_break_fraction"):
            cb = float(np.mean(ss.record.chain_break_fraction))
        if cb is not None:
            chain_breaks.append(float(cb))
        time.sleep(0.2)
    memory = memory_order_parameter(P_S)
    return {
        "memory": float(memory),
        "chain_break_fraction_mean": float(np.mean(chain_breaks)) if chain_breaks else None,
    }


def _schedule(s_p=S_P, t_hold=T_HOLD, t_ramp=5.0):
    return [[0.0, 1.0], [t_ramp, s_p], [t_ramp + t_hold, s_p], [2 * t_ramp + t_hold, 1.0]]


def _run_arm(target, h, J, S, E, auto_scale, chain_strength):
    from dimod import BinaryQuadraticModel  # noqa: E402
    P_S = {}
    chain_breaks = []
    for name, init in _initial_states(S, E).items():
        sampleset = target.sample_ising(
            h, J,
            num_reads=NUM_READS,
            anneal_schedule=_schedule(),
            initial_state=dict(init),
            reinitialize_state=True,
            auto_scale=auto_scale,
            chain_strength=chain_strength,
            answer_mode="raw",
            label=f"spot_embed_as_{'T' if auto_scale else 'F'}_{name}",
        )
        if hasattr(sampleset, "resolve"):
            sampleset.resolve()
        variables = list(sampleset.variables)
        samples = np.asarray(sampleset.record.sample, dtype=np.int8)
        P_S[name] = compute_marginal(samples, S, variables=variables)
        info = dict(sampleset.info or {})
        cb = info.get("embedding_context", {}).get("chain_break_fraction")
        if cb is None and hasattr(sampleset.record, "chain_break_fraction"):
            cb = float(np.mean(sampleset.record.chain_break_fraction))
        if cb is not None:
            chain_breaks.append(float(cb))
        time.sleep(0.2)
    memory = memory_order_parameter(P_S)
    return {
        "memory": float(memory),
        "chain_break_fraction_mean": float(np.mean(chain_breaks)) if chain_breaks else None,
    }


def run(solver: str, output: Path):
    from dwave.system import DWaveSampler, FixedEmbeddingComposite  # noqa: E402
    from minorminer import find_embedding  # noqa: E402

    sampler = DWaveSampler(config_file="./dwave.conf", solver=solver)
    print(f"Solver {sampler.solver.name}: {len(sampler.nodelist)} qubits, {len(sampler.edgelist)} couplers")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        # --- legacy 4-tuple conditions (kept for backwards compatibility) ---
        for label, n_env, lam, W in SPOT_CONDITIONS:
            h, J, S, E = _build_logical(n_env, lam, W)
            logical_edges = list(J.keys())
            print(f"[{label}] n_env={n_env}, lam={lam}, W={W}; logical nodes={len(h)}, edges={len(logical_edges)}")
            embedding = find_embedding(logical_edges, sampler.edgelist, random_seed=DISORDER_SEED)
            if not embedding:
                print("  embedding failed; skipping")
                continue
            chain_lengths = [len(c) for c in embedding.values()]
            print(f"  embedding: {len(embedding)} logical → max chain={max(chain_lengths)}")
            target = FixedEmbeddingComposite(sampler, embedding=embedding)
            res_T = _run_arm(target, h, J, S, E, auto_scale=True, chain_strength=CHAIN_STRENGTH)
            print(f"  auto_scale=True : M={res_T['memory']:.4f}")
            res_F = _run_arm(target, h, J, S, E, auto_scale=False, chain_strength=CHAIN_STRENGTH)
            print(f"  auto_scale=False: M={res_F['memory']:.4f}")
            rows.append({
                "label": label, "n_env": n_env, "coupling_lambda": lam, "disorder_W": W,
                "chain_strength": CHAIN_STRENGTH,
                "max_chain_length": int(max(chain_lengths)),
                "memory_auto_scale_true": res_T["memory"],
                "memory_auto_scale_false": res_F["memory"],
                "delta_memory": res_T["memory"] - res_F["memory"],
                "chain_break_frac_true": res_T["chain_break_fraction_mean"],
                "chain_break_frac_false": res_F["chain_break_fraction_mean"],
            })

        # --- extended 8-tuple conditions: frustrated + UTC default chain strength ---
        for cond in SPOT_CONDITIONS_EX:
            label, n_env, lam, W, mode, subsize, sign_seed, cs_override = cond
            h, J, S, E = _build_logical(n_env, lam, W, subsystem_size=subsize, sign_seed=sign_seed)
            logical_edges = list(J.keys())
            print(f"[{label}] n_env={n_env}, lam={lam}, W={W}, mode={mode}, |S|={subsize}, "
                  f"sign_seed={sign_seed}, cs_override={cs_override}; nodes={len(h)}, edges={len(logical_edges)}")

            use_utc = (cs_override is None)
            if use_utc:
                # EmbeddingComposite picks a fresh embedding each call; OK here
                target_composite = True  # signal to use EmbeddingComposite
                target = None
                chain_strength_value = None  # None -> uniform_torque_compensation
            else:
                embedding = find_embedding(logical_edges, sampler.edgelist, random_seed=DISORDER_SEED)
                if not embedding:
                    print("  embedding failed; skipping")
                    continue
                chain_lengths = [len(c) for c in embedding.values()]
                print(f"  embedding: {len(embedding)} logical → max chain={max(chain_lengths)}")
                target = FixedEmbeddingComposite(sampler, embedding=embedding)
                target_composite = False
                chain_strength_value = cs_override

            if target_composite:
                res_T = _run_arm_with_embedding_composite(sampler, h, J, S, E, auto_scale=True, chain_strength=chain_strength_value)
                # auto_scale=False with UTC default chain strength will typically
                # fail because UTC puts chains above extended_j_range; skip it.
                res_F = {"memory": float("nan"), "chain_break_fraction_mean": None}
            else:
                res_T = _run_arm(target, h, J, S, E, auto_scale=True, chain_strength=chain_strength_value)
                res_F = _run_arm(target, h, J, S, E, auto_scale=False, chain_strength=chain_strength_value)
            mt = res_T['memory']; mf = res_F['memory']
            print(f"  auto_scale=True : M={mt:.4f}, <CB>={res_T['chain_break_fraction_mean']}")
            if isinstance(mf, float) and mf == mf:  # not nan
                print(f"  auto_scale=False: M={mf:.4f}, <CB>={res_F['chain_break_fraction_mean']}")
            else:
                print(f"  auto_scale=False: skipped (UTC chains exceed j_range without rescale)")

            rows.append({
                "label": label, "n_env": n_env, "coupling_lambda": lam, "disorder_W": W,
                "mode": mode, "subsystem_size": subsize, "sign_seed": sign_seed,
                "chain_strength_override": cs_override,
                "memory_auto_scale_true": res_T["memory"],
                "memory_auto_scale_false": res_F["memory"],
                "delta_memory": res_T["memory"] - res_F["memory"],
                "chain_break_frac_true": res_T["chain_break_fraction_mean"],
                "chain_break_frac_false": res_F["chain_break_fraction_mean"],
            })
    finally:
        try:
            sampler.client.close()
        except Exception:
            pass

    summary = {
        "solver": solver, "num_reads": NUM_READS, "s_p": S_P, "t_hold": T_HOLD,
        "chain_strength": CHAIN_STRENGTH, "subsystem_size": SUBSYSTEM_SIZE,
        "rows": rows,
    }
    out_file = output / "spot_check_embedded.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_file}")
    print(f"\nSummary (Δ = M_True − M_False, |Δ| ≤ ?):")
    print(f"{'label':<18} {'max_ch':>6} {'CB_T':>6} {'CB_F':>6} {'M_T':>6} {'M_F':>6} {'Δ M':>7}")
    for r in rows:
        cb_t = r.get('chain_break_frac_true')
        cb_f = r.get('chain_break_frac_false')
        mc = r.get('max_chain_length', 0)
        mt = r.get('memory_auto_scale_true', float('nan'))
        mf = r.get('memory_auto_scale_false', float('nan'))
        dm = r.get('delta_memory', float('nan'))
        print(f"{r['label']:<18} {mc:>6} "
              f"{cb_t if cb_t is not None else float('nan'):>6.3f} "
              f"{cb_f if cb_f is not None else float('nan'):>6.3f} "
              f"{mt:>6.3f} {mf if mf == mf else float('nan'):>6.3f} "
              f"{dm if dm == dm else float('nan'):>+7.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="Advantage2_system1")
    parser.add_argument("--output", default="data/raw/phase4/spot_check_embedded")
    args = parser.parse_args()
    run(args.solver, REPO / args.output)
