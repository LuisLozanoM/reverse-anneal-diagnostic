"""QPU sampling utilities: effective temperature, gauge transforms, reverse anneal.

Forked from paper1/qpu_utils.py. Removed torch dependencies and paper1-specific
code. Pure numpy interface for Ising problems (h, J dicts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import dimod

    _HAS_DIMOD = True
except ImportError:
    dimod = None  # type: ignore[assignment]
    _HAS_DIMOD = False

try:
    from dwave.system.temperatures import (
        fast_effective_temperature,
        maximum_pseudolikelihood_temperature,
    )

    _HAS_TEMPS = True
except ImportError:
    _HAS_TEMPS = False


# ---------------------------------------------------------------------------
# Effective temperature estimate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EffectiveTemperatureEstimate:
    """Estimated effective temperature and its inverse."""

    method: str
    temperature: float
    beta: float
    stderr_temperature: float = 0.0
    stderr_beta: float = 0.0
    num_samples: int = 0

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "method": self.method,
            "temperature": self.temperature,
            "beta": self.beta,
            "stderr_temperature": self.stderr_temperature,
            "stderr_beta": self.stderr_beta,
            "num_samples": self.num_samples,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _temperature_stderr(bootstrap_temperatures: np.ndarray) -> float:
    if bootstrap_temperatures.size == 0:
        return 0.0
    return float(np.sqrt(np.var(bootstrap_temperatures)))


def _beta_from_temperature(temperature: float) -> float:
    if temperature <= 0.0:
        return math.inf
    return float(1.0 / temperature)


def _beta_stderr_from_temperature(
    temperature: float, stderr_temperature: float,
) -> float:
    if temperature <= 0.0:
        return math.inf
    return float(stderr_temperature / (temperature * temperature))


def _get_solver_property(sampler: Any, name: str) -> Any | None:
    """Walk the sampler chain to find a property."""
    stack = [sampler]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        props = getattr(current, "properties", None)
        if isinstance(props, dict) and name in props:
            return props[name]
        child = getattr(current, "child", None)
        if child is not None:
            stack.append(child)
        children = getattr(current, "children", None)
        if children is not None:
            stack.extend(reversed(list(children)))
    return None


def _coerce_samples(
    samples: Any,
    *,
    variables: list[Any] | None,
) -> Any:
    """Coerce samples to (np.ndarray, variables) tuple or dimod.SampleSet."""
    if _HAS_DIMOD and isinstance(samples, dimod.SampleSet):
        return samples

    if isinstance(samples, tuple) and len(samples) == 2:
        return samples

    array = np.asarray(samples, dtype=np.int8)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D sample array, got shape {array.shape}.")
    if variables is None:
        raise ValueError("variables must be provided when samples are not a SampleSet.")
    if len(variables) != array.shape[1]:
        raise ValueError(
            f"Expected {len(variables)} variables, got width {array.shape[1]}."
        )
    return (array, list(variables))


def _coerce_initial_state(
    initial_state: Any, variables: list[Any],
) -> dict[Any, int]:
    """Convert initial state to {variable: spin} dict."""
    if isinstance(initial_state, dict):
        return {v: int(initial_state[v]) for v in variables}
    if isinstance(initial_state, np.ndarray):
        values = initial_state.reshape(-1).tolist()
    else:
        values = list(initial_state)
    if len(values) != len(variables):
        raise ValueError(
            f"Expected {len(variables)} spins, got {len(values)}."
        )
    return {v: int(val) for v, val in zip(variables, values, strict=True)}


def _undo_gauge(sample_set: Any, gauge: dict[Any, int]) -> Any:
    """Undo a gauge transform on a dimod SampleSet."""
    variables = list(sample_set.variables)
    gauge_vector = np.asarray([gauge[v] for v in variables], dtype=np.int8)
    samples = np.asarray(sample_set.record.sample, dtype=np.int8).copy()
    restored = samples * gauge_vector.reshape(1, -1)
    return dimod.SampleSet.from_samples(
        (restored, variables),
        vartype=dimod.SPIN,
        energy=np.asarray(sample_set.record.energy, dtype=float).copy(),
        num_occurrences=np.asarray(sample_set.record.num_occurrences, dtype=np.int64).copy(),
        info=dict(sample_set.info),
    )


# ---------------------------------------------------------------------------
# Effective temperature estimation
# ---------------------------------------------------------------------------


def estimate_mpl_effective_temperature(
    h: dict[Any, float],
    J: dict[tuple[Any, Any], float],
    samples: Any,
    *,
    variables: list[Any] | None = None,
    num_bootstrap_samples: int = 0,
    seed: int | None = None,
    T_guess: float | None = None,
    optimize_method: str | None = "bisect",
    T_bracket: tuple[float, float] = (1e-3, 1000.0),
    sample_weights: np.ndarray | None = None,
) -> EffectiveTemperatureEstimate:
    """Instance-dependent effective temperature via maximum pseudolikelihood."""
    if not _HAS_DIMOD or not _HAS_TEMPS:
        raise ImportError("dimod and dwave.system.temperatures are required.")

    bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
    sample_input = _coerce_samples(samples, variables=variables)

    try:
        temperature, bootstrap_temperatures = maximum_pseudolikelihood_temperature(
            bqm=bqm,
            sampleset=sample_input,
            num_bootstrap_samples=int(num_bootstrap_samples),
            seed=seed,
            T_guess=T_guess,
            optimize_method=optimize_method,
            T_bracket=T_bracket,
            sample_weights=sample_weights,
        )
        resolved_temp = float(temperature)
        stderr_temp = _temperature_stderr(np.asarray(bootstrap_temperatures, dtype=float))
    except ZeroDivisionError:
        resolved_temp = math.inf
        stderr_temp = 0.0

    beta = _beta_from_temperature(resolved_temp)
    stderr_beta = _beta_stderr_from_temperature(resolved_temp, stderr_temp)

    if _HAS_DIMOD and isinstance(sample_input, dimod.SampleSet):
        num_samples = int(np.asarray(sample_input.record.num_occurrences, dtype=np.int64).sum())
    else:
        num_samples = int(np.asarray(sample_input[0]).shape[0])

    return EffectiveTemperatureEstimate(
        method="maximum_pseudolikelihood_temperature",
        temperature=resolved_temp,
        beta=beta,
        stderr_temperature=stderr_temp,
        stderr_beta=stderr_beta,
        num_samples=num_samples,
    )


def estimate_fast_effective_temperature(
    sampler: Any,
    *,
    num_reads: int | None = None,
    seed: int | None = None,
    h_range: tuple[float, float] | None = None,
    nodelist: list[Any] | None = None,
    sampler_params: dict[str, Any] | None = None,
    optimize_method: str | None = "bisect",
    num_bootstrap_samples: int = 0,
) -> EffectiveTemperatureEstimate:
    """Fast sampler-level effective temperature on a selected qubit patch."""
    if not _HAS_TEMPS:
        raise ImportError("dwave.system.temperatures is required.")

    sampler_params = {} if sampler_params is None else dict(sampler_params)
    if h_range is None:
        solver_h_range = _get_solver_property(sampler, "h_range")
        default_probe = (-1.0 / 6.1, 1.0 / 6.1)
        if solver_h_range is None:
            h_range = default_probe
        else:
            clipped_lower = max(float(solver_h_range[0]), default_probe[0])
            clipped_upper = min(float(solver_h_range[1]), default_probe[1])
            h_range = (
                (clipped_lower, clipped_upper)
                if clipped_lower < clipped_upper
                else (float(solver_h_range[0]), float(solver_h_range[1]))
            )

    temperature, stderr_temperature = fast_effective_temperature(
        sampler=sampler,
        num_reads=num_reads,
        seed=seed,
        h_range=h_range,
        nodelist=nodelist,
        sampler_params=sampler_params,
        optimize_method=optimize_method,
        num_bootstrap_samples=int(num_bootstrap_samples),
    )
    resolved_temp = float(temperature)
    resolved_stderr = float(stderr_temperature)
    beta = _beta_from_temperature(resolved_temp)
    stderr_beta = _beta_stderr_from_temperature(resolved_temp, resolved_stderr)

    resolved_num_reads = num_reads
    if resolved_num_reads is None:
        resolved_num_reads = int(sampler_params.get("num_reads", 1000))

    return EffectiveTemperatureEstimate(
        method="fast_effective_temperature",
        temperature=resolved_temp,
        beta=beta,
        stderr_temperature=resolved_stderr,
        stderr_beta=stderr_beta,
        num_samples=int(resolved_num_reads),
    )


# ---------------------------------------------------------------------------
# Gauge transforms
# ---------------------------------------------------------------------------


def apply_gauge_transforms(
    h: dict[Any, float],
    J: dict[tuple[Any, Any], float],
    n_gauges: int,
    *,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate spin-reversal transforms for an Ising problem."""
    if n_gauges <= 0:
        raise ValueError("n_gauges must be positive.")

    variables = tuple(h.keys())
    rng = np.random.default_rng(seed)
    transforms: list[dict[str, Any]] = []

    for idx in range(n_gauges):
        if idx == 0:
            gauge = {v: 1 for v in variables}
        else:
            flips = rng.choice(np.array([-1, 1], dtype=np.int8), size=len(variables))
            gauge = {v: int(f) for v, f in zip(variables, flips, strict=True)}

        transformed_h = {v: float(b) * gauge[v] for v, b in h.items()}
        transformed_J = {
            e: float(c) * gauge[e[0]] * gauge[e[1]] for e, c in J.items()
        }
        transforms.append({"gauge": gauge, "h": transformed_h, "J": transformed_J})

    return transforms


def sample_with_gauges(
    sampler: Any,
    h: dict[Any, float],
    J: dict[tuple[Any, Any], float],
    n_gauges: int,
    num_reads_per_gauge: int,
    *,
    seed: int | None = None,
    max_retries: int = 5,
    retry_delay: float = 10.0,
    **kwargs: Any,
) -> Any:
    """Sample an Ising problem under spin-reversal gauges and concatenate."""
    if not _HAS_DIMOD:
        raise ImportError("dimod is required for gauged sampling.")
    if num_reads_per_gauge <= 0:
        raise ValueError("num_reads_per_gauge must be positive.")

    transforms = apply_gauge_transforms(h=h, J=J, n_gauges=n_gauges, seed=seed)
    variables = list(h.keys())
    samplesets: list[Any] = []

    for gauge_idx, transform in enumerate(transforms):
        sample_kwargs = dict(kwargs)
        if "initial_state" in sample_kwargs:
            state = _coerce_initial_state(sample_kwargs["initial_state"], variables)
            sample_kwargs["initial_state"] = {
                v: state[v] * transform["gauge"][v] for v in variables
            }

        for attempt in range(1, max_retries + 1):
            try:
                ss = sampler.sample_ising(
                    transform["h"], transform["J"],
                    num_reads=int(num_reads_per_gauge),
                    **sample_kwargs,
                )
                if hasattr(ss, "resolve"):
                    ss.resolve()
                samplesets.append(_undo_gauge(ss, transform["gauge"]))
                break
            except Exception as exc:
                if attempt < max_retries:
                    import time
                    time.sleep(retry_delay * attempt)
                else:
                    raise RuntimeError(
                        f"Gauge {gauge_idx} failed after {max_retries} attempts: {exc}"
                    ) from exc

    return dimod.concatenate(samplesets)


# ---------------------------------------------------------------------------
# Reverse anneal sample (low-level)
# ---------------------------------------------------------------------------


def reverse_anneal_sample(
    sampler: Any,
    h: dict[Any, float],
    J: dict[tuple[Any, Any], float],
    initial_state: Any,
    schedule: list[list[float]],
    num_reads: int,
    **kwargs: Any,
) -> Any:
    """Run a reverse-anneal call from a fixed initial state."""
    if num_reads <= 0:
        raise ValueError("num_reads must be positive.")

    variables = list(h.keys())
    sample_kwargs = dict(kwargs)
    sample_kwargs["anneal_schedule"] = schedule
    sample_kwargs.setdefault("reinitialize_state", True)
    sample_kwargs["initial_state"] = _coerce_initial_state(initial_state, variables)
    return sampler.sample_ising(h, J, num_reads=int(num_reads), **sample_kwargs)
