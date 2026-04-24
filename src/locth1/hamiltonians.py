"""Ising Hamiltonian builders for subsystem thermalization experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import networkx as nx

    _HAS_NX = True
except ImportError:
    nx = None  # type: ignore[assignment]
    _HAS_NX = False

try:
    import dwave_networkx as dnx

    _HAS_DWAVE_NX = True
except ImportError:
    dnx = None  # type: ignore[assignment]
    _HAS_DWAVE_NX = False


def _require_dwave_nx() -> None:
    if not _HAS_DWAVE_NX:
        raise ImportError("dwave_networkx is required for native-graph construction.")


# ---------------------------------------------------------------------------
# Native hardware graphs
# ---------------------------------------------------------------------------


def _make_native_hardware_graph(topology: str, n_qubits: int) -> nx.Graph:
    """Build a connected subgraph of the hardware topology."""
    _require_dwave_nx()
    if topology == "zephyr":
        m = max(2, int(np.ceil(np.sqrt(n_qubits / 48))))
        while True:
            full = dnx.zephyr_graph(m)
            if full.number_of_nodes() >= n_qubits:
                break
            m += 1
    elif topology == "pegasus":
        m = max(2, int(np.ceil(np.sqrt(n_qubits / 24))))
        while True:
            full = dnx.pegasus_graph(m)
            if full.number_of_nodes() >= n_qubits:
                break
            m += 1
    else:
        raise ValueError(f"Unknown native topology: {topology!r}")

    # Take a connected subgraph of the requested size via BFS
    nodes = list(full.nodes())
    start = nodes[0]
    visited: list[int] = []
    queue = [start]
    seen = {start}
    while queue and len(visited) < n_qubits:
        node = queue.pop(0)
        visited.append(node)
        for nbr in full.neighbors(node):
            if nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)

    return full.subgraph(visited[:n_qubits]).copy()


def build_native_graph(
    topology: str,
    n_qubits: int,
    subsystem_size: int,
    coupling_lambda: float,
    disorder_W: float,
    seed: int,
) -> dict[str, Any]:
    """Build an Ising problem on a native hardware graph with S/E partition."""
    graph = _make_native_hardware_graph(topology, n_qubits)
    if graph.number_of_nodes() < n_qubits:
        raise ValueError(
            f"Could not extract {n_qubits} connected qubits from {topology} topology."
        )

    S_qubits, E_qubits = partition_subsystem(
        graph, strategy="geometric", subsystem_size=subsystem_size, seed=seed,
    )

    rng = np.random.default_rng(seed)
    h: dict[int, float] = {}
    for node in graph.nodes():
        h[node] = float(rng.uniform(-disorder_W, disorder_W))

    S_set = set(S_qubits)
    E_set = set(E_qubits)
    J: dict[tuple[int, int], float] = {}
    edges: list[tuple[int, int]] = []
    for u, v in graph.edges():
        edges.append((u, v))
        u_in_S = u in S_set
        v_in_S = v in S_set
        # Boundary edge: one in S, one in E
        if u_in_S != v_in_S:
            J[(u, v)] = -1.0 * coupling_lambda
        else:
            J[(u, v)] = -1.0

    return {
        "h": h,
        "J": J,
        "S_qubits": S_qubits,
        "E_qubits": E_qubits,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# Logical graphs
# ---------------------------------------------------------------------------


def _make_logical_graph(graph_type: str, n_qubits: int, seed: int) -> nx.Graph:
    """Build a logical (non-hardware) graph."""
    if graph_type == "random_regular":
        degree = min(3, n_qubits - 1)
        return nx.random_regular_graph(degree, n_qubits, seed=seed)
    elif graph_type == "2d_lattice":
        side = int(np.ceil(np.sqrt(n_qubits)))
        G = nx.grid_2d_graph(side, side)
        # Relabel to integer nodes and truncate
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        nodes_to_keep = list(range(n_qubits))
        return G.subgraph(nodes_to_keep).copy()
    elif graph_type == "complete":
        return nx.complete_graph(n_qubits)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type!r}")


def build_logical_graph(
    graph_type: str,
    n_qubits: int,
    subsystem_size: int,
    coupling_lambda: float,
    disorder_W: float,
    seed: int,
) -> dict[str, Any]:
    """Build an Ising problem on a logical graph with S/E partition."""
    graph = _make_logical_graph(graph_type, n_qubits, seed)

    S_qubits, E_qubits = partition_subsystem(
        graph, strategy="geometric", subsystem_size=subsystem_size, seed=seed,
    )

    rng = np.random.default_rng(seed)
    h: dict[int, float] = {}
    for node in graph.nodes():
        h[node] = float(rng.uniform(-disorder_W, disorder_W))

    S_set = set(S_qubits)
    J: dict[tuple[int, int], float] = {}
    edges: list[tuple[int, int]] = []
    for u, v in graph.edges():
        edges.append((u, v))
        u_in_S = u in S_set
        v_in_S = v in S_set
        if u_in_S != v_in_S:
            J[(u, v)] = -1.0 * coupling_lambda
        else:
            J[(u, v)] = -1.0

    return {
        "h": h,
        "J": J,
        "S_qubits": S_qubits,
        "E_qubits": E_qubits,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# Subsystem partitioning
# ---------------------------------------------------------------------------


def partition_subsystem(
    graph: nx.Graph,
    strategy: str,
    subsystem_size: int,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Partition graph into subsystem S and environment E."""
    nodes = list(graph.nodes())
    if subsystem_size >= len(nodes):
        raise ValueError(
            f"subsystem_size ({subsystem_size}) must be < number of nodes ({len(nodes)})."
        )

    rng = np.random.default_rng(seed)

    if strategy == "geometric":
        S_qubits = _partition_geometric(graph, subsystem_size, rng)
    elif strategy == "random":
        S_qubits = _partition_random(graph, subsystem_size, rng)
    elif strategy == "max_connectivity":
        S_qubits = _partition_max_connectivity(graph, subsystem_size, rng)
    else:
        raise ValueError(f"Unknown partition strategy: {strategy!r}")

    S_set = set(S_qubits)
    E_qubits = [n for n in nodes if n not in S_set]
    return S_qubits, E_qubits


def _partition_geometric(
    graph: nx.Graph, size: int, rng: np.random.Generator,
) -> list[int]:
    """BFS ball from a random seed node."""
    nodes = list(graph.nodes())
    start = nodes[int(rng.integers(len(nodes)))]
    visited: list[int] = []
    queue = [start]
    seen = {start}
    while queue and len(visited) < size:
        node = queue.pop(0)
        visited.append(node)
        neighbors = list(graph.neighbors(node))
        rng.shuffle(neighbors)
        for nbr in neighbors:
            if nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)
    return visited[:size]


def _partition_random(
    graph: nx.Graph, size: int, rng: np.random.Generator,
) -> list[int]:
    """Random connected subgraph via random walk."""
    nodes = list(graph.nodes())
    start = nodes[int(rng.integers(len(nodes)))]
    visited = [start]
    visited_set = {start}
    current = start
    while len(visited) < size:
        neighbors = [n for n in graph.neighbors(current) if n not in visited_set]
        if not neighbors:
            # Backtrack: pick any visited node with unvisited neighbors
            candidates = [
                v for v in visited
                if any(n not in visited_set for n in graph.neighbors(v))
            ]
            if not candidates:
                break
            current = candidates[int(rng.integers(len(candidates)))]
            continue
        current = neighbors[int(rng.integers(len(neighbors)))]
        visited.append(current)
        visited_set.add(current)
    return visited[:size]


def _partition_max_connectivity(
    graph: nx.Graph, size: int, rng: np.random.Generator,
) -> list[int]:
    """Greedy: maximize boundary edges between S and E."""
    nodes = list(graph.nodes())
    # Start from the highest-degree node
    degrees = dict(graph.degree())
    start = max(nodes, key=lambda n: degrees[n])

    selected = [start]
    selected_set = {start}

    while len(selected) < size:
        best_node = None
        best_boundary = -1
        # Consider all neighbors of current selection
        candidates = set()
        for s in selected:
            for nbr in graph.neighbors(s):
                if nbr not in selected_set:
                    candidates.add(nbr)
        if not candidates:
            break
        for c in candidates:
            # Count boundary edges if we add c
            trial = selected_set | {c}
            boundary = sum(
                1 for u, v in graph.edges(trial)
                if (u in trial) != (v in trial)
            )
            if boundary > best_boundary:
                best_boundary = boundary
                best_node = c
        if best_node is None:
            break
        selected.append(best_node)
        selected_set.add(best_node)

    return selected[:size]
