from contextlib import contextmanager
from typing import Iterable, Tuple

import networkx as nx

from src.utils.typing import ConversionPath


def path_cost(path: ConversionPath) -> float:
    """Calculate the total cost of a path."""
    return sum(p.cost for p in path)


@contextmanager
def without_edges(g: nx.Graph, edges: Iterable[Tuple[type, type]] = None):
    held = {}
    for a, b in edges or []:
        held[(a, b)] = g.adj[a][b]
        g.remove_edge(a, b)

    try:
        yield g
    finally:
        for (a, b), kwargs in held.items():
            g.add_edge(a, b, **kwargs)
