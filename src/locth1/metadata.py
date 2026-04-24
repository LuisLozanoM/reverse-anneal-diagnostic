"""Job metadata logging and raw result I/O (JSON + HDF5)."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py

    _HAS_H5PY = True
except ImportError:
    h5py = None  # type: ignore[assignment]
    _HAS_H5PY = False


# ---------------------------------------------------------------------------
# Metadata dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class JobMetadata:
    """Full metadata for a single QPU job."""

    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    solver: str = ""
    graph_id: str | None = None
    anneal_schedule: list[tuple[float, float]] | None = None
    initial_state_label: str = ""
    h: dict[Any, float] = field(default_factory=dict)
    J: dict[tuple[Any, Any], float] = field(default_factory=dict)
    disorder_realization_seed: int = 0
    num_reads: int = 0
    timing: dict[str, float] = field(default_factory=dict)
    calibration_epoch: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _serialize_key(k: Any) -> str:
    """Convert tuple or other keys to JSON-safe string."""
    if isinstance(k, tuple):
        return str(k)
    return str(k)


def _serialize_value(v: Any) -> Any:
    """Make values JSON-serializable."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {_serialize_key(kk): _serialize_value(vv) for kk, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialize_value(x) for x in v]
    return v


def _metadata_to_dict(m: JobMetadata) -> dict[str, Any]:
    """Convert metadata to a JSON-safe dict."""
    d = asdict(m)
    return _serialize_value(d)


def _parse_ising_key(s: str) -> Any:
    """Parse a string key back to int or tuple of ints."""
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        parts = s[1:-1].split(",")
        return tuple(int(p.strip()) for p in parts if p.strip())
    try:
        return int(s)
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def save_metadata(metadata: JobMetadata, path: Path) -> None:
    """Save job metadata to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_metadata_to_dict(metadata), f, indent=2)


def load_metadata(path: Path) -> JobMetadata:
    """Load job metadata from a JSON file."""
    with open(path) as f:
        d = json.load(f)

    # Reconstruct h and J with proper key types
    h_raw = d.get("h", {})
    h = {_parse_ising_key(k): float(v) for k, v in h_raw.items()}

    J_raw = d.get("J", {})
    J = {_parse_ising_key(k): float(v) for k, v in J_raw.items()}

    schedule = d.get("anneal_schedule")
    if schedule is not None:
        schedule = [tuple(pt) for pt in schedule]

    return JobMetadata(
        job_id=d.get("job_id", ""),
        timestamp=d.get("timestamp", ""),
        solver=d.get("solver", ""),
        graph_id=d.get("graph_id"),
        anneal_schedule=schedule,
        initial_state_label=d.get("initial_state_label", ""),
        h=h,
        J=J,
        disorder_realization_seed=int(d.get("disorder_realization_seed", 0)),
        num_reads=int(d.get("num_reads", 0)),
        timing=d.get("timing", {}),
        calibration_epoch=d.get("calibration_epoch"),
        extras=d.get("extras", {}),
    )


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------


def save_raw_results(
    samples: np.ndarray,
    energies: np.ndarray,
    metadata: JobMetadata | dict[str, Any],
    path: Path,
) -> None:
    """Save raw QPU results to HDF5: samples as int8, energies as float64."""
    if not _HAS_H5PY:
        raise ImportError("h5py is required for HDF5 I/O.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    samples_arr = np.asarray(samples, dtype=np.int8)
    energies_arr = np.asarray(energies, dtype=np.float64)

    if isinstance(metadata, dict):
        meta_dict = _serialize_value(metadata)
    else:
        meta_dict = _metadata_to_dict(metadata)

    with h5py.File(path, "w") as f:
        f.create_dataset("samples", data=samples_arr, compression="gzip")
        f.create_dataset("energies", data=energies_arr, compression="gzip")
        # Store metadata as attributes
        for k, v in meta_dict.items():
            try:
                if isinstance(v, (dict, list)):
                    f.attrs[k] = json.dumps(v)
                elif v is None:
                    f.attrs[k] = "null"
                else:
                    f.attrs[k] = v
            except TypeError:
                f.attrs[k] = str(v)


def load_raw_results(path: Path) -> dict[str, Any]:
    """Load raw QPU results from HDF5."""
    if not _HAS_H5PY:
        raise ImportError("h5py is required for HDF5 I/O.")

    with h5py.File(path, "r") as f:
        samples = np.asarray(f["samples"], dtype=np.int8)
        energies = np.asarray(f["energies"], dtype=np.float64)

        # Reconstruct metadata dict from attrs
        meta: dict[str, Any] = {}
        for k, v in f.attrs.items():
            if isinstance(v, str):
                if v == "null":
                    meta[k] = None
                else:
                    try:
                        meta[k] = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        meta[k] = v
            else:
                meta[k] = v

    return {
        "samples": samples,
        "energies": energies,
        "metadata": meta,
    }
