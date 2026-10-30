"""Exact canonical keys for tiny colored directed relation graphs.

Node IDs identify actual entities, not their changing goal roles. Colors encode
roles and capabilities. This module proves symbolic equivalence only; it makes
no assertion that policy transition probabilities are invariant.
"""

from dataclasses import dataclass
from itertools import permutations, product
import json
from math import factorial
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ColoredGraph:
    nodes: Mapping[str, str]
    edges: tuple[tuple[str, str, str], ...]  # (source_id, relation, target_id)

    def __post_init__(self):
        nodes = dict(self.nodes)
        if any(not isinstance(k, str) or not isinstance(v, str)
               for k, v in nodes.items()):
            raise TypeError("Node IDs and colors must be strings")
        edges = tuple(tuple(e) for e in self.edges)
        for edge in edges:
            if len(edge) != 3:
                raise ValueError("Edges are (source, relation, target) triples")
            source, relation, target = edge
            if source not in nodes or target not in nodes:
                raise ValueError("Every edge endpoint must be a graph node")
            if not isinstance(relation, str):
                raise TypeError("Relation colors must be strings")
        if len(set(edges)) != len(edges):
            raise ValueError("Duplicate relation triples are not permitted")
        object.__setattr__(self, "nodes", MappingProxyType(nodes))
        object.__setattr__(self, "edges", tuple(sorted(edges)))


@dataclass(frozen=True)
class CanonicalForm:
    key: str
    to_canonical: Mapping[str, int]


def canonicalize(graph: ColoredGraph, *, max_permutations: int = 10_000
                 ) -> CanonicalForm:
    """Enumerate permutations within color classes; fail before a large search.

    The key is invariant to node-ID renaming. In a symmetric graph the returned
    permutation is one valid canonical mapping, not a unique physical identity.
    Preserve physical identities explicitly for transitions.
    """
    if max_permutations < 1:
        raise ValueError("max_permutations must be positive")
    groups: dict[str, list[str]] = {}
    for node, color in graph.nodes.items():
        groups.setdefault(color, []).append(node)
    colors = sorted(groups)
    search_size = 1
    for color in colors:
        groups[color].sort()
        search_size *= factorial(len(groups[color]))
    if search_size > max_permutations:
        raise ValueError(
            f"Canonical search requires {search_size} permutations; "
            f"limit is {max_permutations}. Refine colors or narrow the graph."
        )
    canonical_colors = tuple(
        color for color in colors for _ in groups[color]
    )
    best_edges = None
    best_map = None
    for ordering in product(*(permutations(groups[c]) for c in colors)):
        ordered_nodes = tuple(node for group in ordering for node in group)
        mapping = {node: i for i, node in enumerate(ordered_nodes)}
        edges = tuple(sorted(
            (mapping[source], relation, mapping[target])
            for source, relation, target in graph.edges
        ))
        if best_edges is None or edges < best_edges:
            best_edges, best_map = edges, mapping
    key = json.dumps([canonical_colors, best_edges],
                     ensure_ascii=False, separators=(",", ":"))
    return CanonicalForm(key, MappingProxyType(best_map or {}))


@dataclass(frozen=True)
class TransitionKey:
    source_key: str
    joint_key: str


def canonical_transition(
    before: ColoredGraph,
    after: ColoredGraph,
    *,
    identity: Mapping[str, str] | None = None,
    max_permutations: int = 10_000,
) -> TransitionKey:
    """Key a before/after graph with explicit cross-time physical identity.

    With stable environment entity IDs, omitted identity maps shared IDs to
    themselves. If the adapter renames IDs, supply an injective correspondence.
    Goal-witness changes must NOT be represented as physical identity.

    Pool candidate donors by source_key, which never uses the future. joint_key
    describes an observed outgoing event; it must not select which future
    outcomes are retained in the source row.
    """
    if identity is None:
        identity = {node: node for node in before.nodes if node in after.nodes}
    identity = dict(identity)
    if not set(identity).issubset(before.nodes):
        raise ValueError("Identity source is absent from the before graph")
    if not set(identity.values()).issubset(after.nodes):
        raise ValueError("Identity target is absent from the after graph")
    if len(set(identity.values())) != len(identity):
        raise ValueError("Cross-time identity must be injective")

    def node_id(layer: str, node: str) -> str:
        return json.dumps([layer, node], ensure_ascii=False)

    nodes = {}
    edges = []
    for layer, graph in (("before", before), ("after", after)):
        for node, color in graph.nodes.items():
            nodes[node_id(layer, node)] = json.dumps([layer, color])
        for source, relation, target in graph.edges:
            edges.append((node_id(layer, source),
                          json.dumps(["relation", layer, relation]),
                          node_id(layer, target)))
    for old, new in identity.items():
        edges.append((node_id("before", old), "CROSS_TIME_IDENTITY",
                      node_id("after", new)))
    return TransitionKey(
        source_key=canonicalize(before,
                                max_permutations=max_permutations).key,
        joint_key=canonicalize(ColoredGraph(nodes, tuple(edges)),
                               max_permutations=max_permutations).key,
    )
