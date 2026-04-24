"""Minor embedding utilities for cross-QPU Ising experiments."""

from __future__ import annotations

from typing import Any

try:
    import networkx as nx

    _HAS_NX = True
except ImportError:
    nx = None  # type: ignore[assignment]
    _HAS_NX = False

try:
    import minorminer

    _HAS_MINORMINER = True
except ImportError:
    minorminer = None  # type: ignore[assignment]
    _HAS_MINORMINER = False

try:
    import dwave.embedding as dwave_embedding

    _HAS_DWAVE_EMBED = True
except ImportError:
    dwave_embedding = None  # type: ignore[assignment]
    _HAS_DWAVE_EMBED = False

try:
    import dwave_networkx as dnx

    _HAS_DWAVE_NX = True
except ImportError:
    dnx = None  # type: ignore[assignment]
    _HAS_DWAVE_NX = False


# ---------------------------------------------------------------------------
# Embedding search
# ---------------------------------------------------------------------------


def find_embedding(
    source_graph: nx.Graph,
    target_sampler: Any,
    *,
    max_chain_length: int | None = None,
    seed: int | None = None,
) -> dict[Any, list[Any]]:
    """Find a minor embedding of source_graph into the target sampler topology."""
    if not _HAS_MINORMINER:
        raise ImportError("minorminer is required for embedding search.")

    target_edges = list(target_sampler.edgelist)
    kwargs: dict[str, Any] = {}
    if seed is not None:
        kwargs["random_seed"] = seed
    if max_chain_length is not None:
        kwargs["max_no_improvement"] = 50
        # minorminer doesn't directly cap chain length, but we filter afterward

    embedding = minorminer.find_embedding(
        list(source_graph.edges()), target_edges, **kwargs,
    )

    if not embedding:
        raise RuntimeError("minorminer failed to find an embedding.")

    if max_chain_length is not None:
        long_chains = {v: c for v, c in embedding.items() if len(c) > max_chain_length}
        if long_chains:
            worst = max(len(c) for c in long_chains.values())
            raise RuntimeError(
                f"Embedding has chains up to length {worst}, "
                f"exceeding max_chain_length={max_chain_length}."
            )

    return embedding


# ---------------------------------------------------------------------------
# Ising embedding
# ---------------------------------------------------------------------------


def embed_ising(
    h: dict[Any, float],
    J: dict[tuple[Any, Any], float],
    embedding: dict[Any, list[Any]],
    target_adjacency: dict[Any, set[Any]] | nx.Graph,
) -> tuple[dict[Any, float], dict[tuple[Any, Any], float], dict[Any, Any]]:
    """Embed an Ising problem using a precomputed embedding."""
    if not _HAS_DWAVE_EMBED:
        raise ImportError("dwave.embedding is required.")

    if isinstance(target_adjacency, nx.Graph):
        adj = {n: set(target_adjacency.neighbors(n)) for n in target_adjacency.nodes()}
    else:
        adj = target_adjacency

    embedded_h, embedded_J = dwave_embedding.embed_ising(h, J, embedding, adj)

    # Build chain-to-variable map
    chain_to_var: dict[Any, Any] = {}
    for var, chain in embedding.items():
        for qubit in chain:
            chain_to_var[qubit] = var

    return embedded_h, embedded_J, chain_to_var


# ---------------------------------------------------------------------------
# Cross-QPU common subgraph
# ---------------------------------------------------------------------------


def _hardware_graph(topology: str, n_qubits: int) -> nx.Graph:
    """Build a hardware graph for the given topology, large enough for n_qubits."""
    if not _HAS_DWAVE_NX:
        raise ImportError("dwave_networkx is required.")
    if topology == "zephyr":
        m = max(2, int(n_qubits / 48) + 1)
        while True:
            G = dnx.zephyr_graph(m)
            if G.number_of_nodes() >= n_qubits:
                return G
            m += 1
    elif topology == "pegasus":
        m = max(2, int(n_qubits / 24) + 1)
        while True:
            G = dnx.pegasus_graph(m)
            if G.number_of_nodes() >= n_qubits:
                return G
            m += 1
    else:
        raise ValueError(f"Unknown topology: {topology!r}")


def common_subgraph(
    topology_a: str,
    topology_b: str,
    n_qubits: int,
) -> nx.Graph:
    """Find a subgraph natively embeddable on both topologies.

    Builds a complete graph on n_qubits nodes, then finds the largest
    subset that can be natively embedded (chain length 1) on both targets.
    In practice, this finds a common clique or near-clique.
    """
    if not _HAS_MINORMINER:
        raise ImportError("minorminer is required.")

    G_a = _hardware_graph(topology_a, n_qubits * 4)
    G_b = _hardware_graph(topology_b, n_qubits * 4)

    # Start with a complete graph and shrink until it embeds on both
    for size in range(n_qubits, 0, -1):
        source = nx.complete_graph(size)
        source_edges = list(source.edges())
        try:
            emb_a = minorminer.find_embedding(source_edges, list(G_a.edges()))
            if not emb_a or any(len(c) > 1 for c in emb_a.values()):
                continue
            emb_b = minorminer.find_embedding(source_edges, list(G_b.edges()))
            if not emb_b or any(len(c) > 1 for c in emb_b.values()):
                continue
            return source
        except Exception:
            continue

    # Fallback: try path graphs
    for size in range(n_qubits, 0, -1):
        source = nx.path_graph(size)
        source_edges = list(source.edges())
        try:
            emb_a = minorminer.find_embedding(source_edges, list(G_a.edges()))
            if not emb_a or any(len(c) > 1 for c in emb_a.values()):
                continue
            emb_b = minorminer.find_embedding(source_edges, list(G_b.edges()))
            if not emb_b or any(len(c) > 1 for c in emb_b.values()):
                continue
            return source
        except Exception:
            continue

    raise RuntimeError(
        f"Could not find a common subgraph of size >= 1 for "
        f"{topology_a} and {topology_b}."
    )
