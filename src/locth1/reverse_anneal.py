"""Reverse annealing driver for raw Ising problems (h, J dicts)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from dwave.system import DWaveSampler, FixedEmbeddingComposite

    _HAS_DWAVE = True
except ImportError:
    DWaveSampler = None  # type: ignore[assignment,misc]
    FixedEmbeddingComposite = None  # type: ignore[assignment,misc]
    _HAS_DWAVE = False

_DEFAULT_CONFIG = "./dwave.conf"


def _require_dwave() -> None:
    if not _HAS_DWAVE:
        raise ImportError("dwave-system is required for QPU access.")


def _resolve_config(config_file: str | None) -> dict[str, Any]:
    """Build config kwargs for DWaveSampler."""
    if config_file is not None:
        return {"config_file": config_file}
    if Path(_DEFAULT_CONFIG).exists():
        return {"config_file": _DEFAULT_CONFIG}
    return {}


def _extract_timing(sampleset) -> dict[str, float]:
    """Extract QPU timing from sampleset.info."""
    info = getattr(sampleset, "info", {}) or {}
    timing = info.get("timing", {}) or {}
    return {k: float(v) for k, v in timing.items() if isinstance(v, (int, float))}


def _sampler_metadata(sampler) -> dict[str, Any]:
    """Extract solver identity and hardware metadata."""
    props = getattr(sampler, "properties", {}) or {}
    topo = props.get("topology", {}) if isinstance(props.get("topology"), dict) else {}
    return {
        "solver": props.get("chip_id", "unknown"),
        "graph_id": props.get("graph_id") or props.get("chip_id"),
        "calibration_epoch": props.get("calibration_epoch"),
        "topology_family": topo.get("type"),
    }


# ---------------------------------------------------------------------------
# Schedule builder
# ---------------------------------------------------------------------------


def build_anneal_schedule(
    s_p: float,
    t_hold: float,
    t_ramp_down: float = 5.0,
    t_ramp_up: float = 5.0,
) -> list[tuple[float, float]]:
    """Build a four-point reverse anneal schedule.

    Returns [(0, 1.0), (t_down, s_p), (t_down+t_hold, s_p), (t_down+t_hold+t_up, 1.0)].
    """
    if not 0.0 <= s_p <= 1.0:
        raise ValueError(f"s_p must be in [0, 1], got {s_p}")
    if t_hold < 0 or t_ramp_down < 0 or t_ramp_up < 0:
        raise ValueError("Time parameters must be non-negative.")
    t1 = t_ramp_down
    t2 = t1 + t_hold
    t3 = t2 + t_ramp_up
    return [(0.0, 1.0), (t1, s_p), (t2, s_p), (t3, 1.0)]


# ---------------------------------------------------------------------------
# Core reverse-anneal runner
# ---------------------------------------------------------------------------


def run_reverse_anneal(
    h: dict[Any, float],
    J: dict[tuple[Any, Any], float],
    initial_state: dict[Any, int],
    s_p: float,
    t_hold: float,
    num_reads: int,
    solver: str,
    *,
    t_ramp_down: float = 5.0,
    t_ramp_up: float = 5.0,
    reinitialize: bool = True,
    config_file: str | None = None,
    sampler: Any | None = None,
    embedding: dict[Any, list[Any]] | None = None,
    auto_embed: bool = True,
    auto_scale: bool = False,
    answer_mode: str = "raw",
    label: str | None = None,
) -> dict[str, Any]:
    """Run a single reverse-anneal job and return structured results.

    When auto_embed=True (default) and no explicit embedding is given,
    wraps the sampler with EmbeddingComposite for automatic minor embedding.
    Set auto_embed=False when h/J use physical QPU qubit labels.

    auto_scale is forced False by default so that the programmed (h, J) are
    exactly what the user supplied.  Passing auto_scale=True is permitted but
    should only be used when the caller has independently verified that the
    problem already fits within the solver's programmable ranges, since
    D-Wave's default rescaling silently changes the Hamiltonian and breaks
    quantitative comparisons with classical baselines.
    """
    _require_dwave()
    from dwave.system import EmbeddingComposite

    schedule = build_anneal_schedule(s_p, t_hold, t_ramp_down, t_ramp_up)
    owns_sampler = sampler is None

    try:
        if sampler is None:
            config = _resolve_config(config_file)
            config["solver"] = solver
            sampler = DWaveSampler(**config)

        # Wrap with embedding composite
        if embedding is not None:
            target = FixedEmbeddingComposite(sampler, embedding=embedding)
        elif auto_embed:
            target = EmbeddingComposite(sampler)
        else:
            target = sampler

        sample_kwargs: dict[str, Any] = {
            "num_reads": num_reads,
            "anneal_schedule": [list(pt) for pt in schedule],
            "initial_state": dict(initial_state),
            "reinitialize_state": reinitialize,
            "auto_scale": auto_scale,
            "answer_mode": answer_mode,
        }
        if label is not None:
            sample_kwargs["label"] = label

        sampleset = target.sample_ising(h, J, **sample_kwargs)
        # Force resolution of lazy sampleset
        if hasattr(sampleset, "resolve"):
            sampleset.resolve()

        variables = list(sampleset.variables)
        samples = np.asarray(sampleset.record.sample, dtype=np.int8)
        energies = np.asarray(sampleset.record.energy, dtype=np.float64)
        timing = _extract_timing(sampleset)
        meta = _sampler_metadata(sampler)

        info_dict = dict(getattr(sampleset, "info", {}) or {})
        embedding_context = info_dict.get("embedding_context")
        if embedding_context is None and hasattr(sampleset.record, "chain_break_fraction"):
            embedding_context = {
                "chain_break_fraction_mean": float(np.mean(sampleset.record.chain_break_fraction)),
            }
        problem_label = info_dict.get("problem_label") or info_dict.get("label")

        return {
            "samples": samples,
            "energies": energies,
            "variables": variables,
            "timing": timing,
            "metadata": {
                "solver": meta["solver"],
                "graph_id": meta["graph_id"],
                "embedding_context": embedding_context,
                "auto_scale": auto_scale,
                "problem_label": problem_label,
                "calibration_epoch": meta["calibration_epoch"],
                "anneal_schedule": schedule,
                "initial_state": dict(initial_state),
            },
        }
    finally:
        if owns_sampler and sampler is not None:
            try:
                sampler.client.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_batch(
    configs: list[dict[str, Any]],
    *,
    rate_limit_delay: float = 1.0,
) -> list[dict[str, Any]]:
    """Run multiple reverse-anneal configs sequentially with rate limiting."""
    results: list[dict[str, Any]] = []
    for i, config in enumerate(configs):
        result = run_reverse_anneal(**config)
        results.append(result)
        if i < len(configs) - 1 and rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
    return results


# ---------------------------------------------------------------------------
# Solver properties
# ---------------------------------------------------------------------------


def get_solver_properties(
    solver: str,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Query key solver properties for experiment planning."""
    _require_dwave()
    config = _resolve_config(config_file)
    config["solver"] = solver
    sampler = DWaveSampler(**config)
    try:
        props = sampler.properties
        topo = props.get("topology", {})
        return {
            "max_anneal_schedule_points": props.get("max_anneal_schedule_points"),
            "annealing_time_range": props.get("annealing_time_range"),
            "h_range": props.get("h_range"),
            "j_range": props.get("extended_j_range", props.get("j_range")),
            "num_reads_range": props.get("num_reads_range"),
            "topology": topo.get("type") if isinstance(topo, dict) else None,
            "num_qubits": props.get("num_active_qubits", len(list(sampler.nodelist))),
            "num_couplers": props.get("num_active_couplers", len(list(sampler.edgelist))),
        }
    finally:
        try:
            sampler.client.close()
        except Exception:
            pass
