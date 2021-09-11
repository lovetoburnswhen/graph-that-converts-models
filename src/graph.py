import itertools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator
from warnings import warn

import networkx as nx
from devtools import debug
from networkx.exception import NetworkXNoPath

from src.utils.iter import IterProxy
from src.utils.typing import ConversionPath


class FailedConversionWarning(UserWarning):
    def __init__(self, src, dest, exc):
        self.src = src
        self.dest = dest
        self.exc = exc

    def __str__(self):
        return (
            f'Failed on {self.src.__name__} -> {self.dest.__name__}. '
            f'Working around\nError message:\n{self.exc}'
        )


@dataclass
class ConversionEdge:
    source: type
    target: type
    func: Callable
    cost: float


class ConversionGraph:
    def __init__(self, name: str):
        self.name = name
        # self._graph = nx.MultiDiGraph(name=name)
        self._graph = nx.DiGraph(name=name)

    def register(self, source, target, cost=1.0):
        """Decorator that registers a conversion fn between 2 types

        Args:
            source: The type of the object being converted
            target: The type to convert to
        """
        # https://github.com/blaze/odo/blob/master/odo/core.py#L70

        def _wrapper(func: Callable):
            self._graph.add_edge(source, target, cost=cost, func=func)
            return func

        return _wrapper

    def path(self, *args, **kwargs) -> Iterator[ConversionEdge]:
        return path(self._graph, *args, **kwargs)

    def __call__(self, source, target, **kwargs):
        return _transform(self._graph, source=source, target=target, **kwargs)


def _transform(
    graph, target, source, excluded_edges=None, raise_on_errors: bool = False, **kwargs
):
    """Transform source to target type using graph of transformations"""
    # take a copy so we can mutate without affecting the input
    excluded_edges = excluded_edges.copy() if excluded_edges is not None else set()

    pth = path(graph, source=type(source), target=target, excluded_edges=excluded_edges)

    x = source
    path_proxy: IterProxy[ConversionEdge] = IterProxy(pth)
    for edge in path_proxy:
        try:
            x = edge.func(x, excluded_edges=excluded_edges, **kwargs)

        except NotImplementedError as e:
            if raise_on_errors:
                raise
            warn(FailedConversionWarning(edge.source, edge.target, e))

            # exclude the broken edge
            excluded_edges |= {(edge.source, edge.target)}

            # compute the path from `source` to `target` excluding
            # the edge that broke
            fresh_path = list(
                path(graph, type(source), target, excluded_edges=excluded_edges)
            )
            fresh_path_cost = path_cost(fresh_path)

            # compute the path from the current `source` type
            # to the `target`
            try:
                greedy_path = list(
                    path(graph, edge.target, target, excluded_edges=excluded_edges)
                )
            except NetworkXNoPath:
                greedy_path_cost = float("inf")
            else:
                greedy_path_cost = path_cost(greedy_path)

            if fresh_path_cost < greedy_path_cost:
                # it is faster to start over from `source` with a new path
                x = source
                pth = fresh_path
            else:
                # it is faster to work around our broken edge from our
                # current location
                pth = greedy_path

            path_proxy.it = pth

    return x


_virtual_superclasses = (Iterator,)


def path(
    graph: nx.DiGraph, source, target, excluded_edges=None
) -> Iterator[ConversionEdge]:
    """Path of functions between two types"""
    debug(source, target, graph.edges)
    if not isinstance(source, type):
        source = type(source)
    if not isinstance(target, type):
        target = type(target)

    for cls in itertools.chain(source.mro(), _virtual_superclasses):
        if cls in graph:
            source = cls
            break
    else:
        raise ValueError(
            f'Neither {source=} nor any of it\'s parent classes are in {graph=}'
        )

    with without_edges(graph, excluded_edges) as g:
        shortest_pth = nx.shortest_path(g, source=source, target=target, weight='cost')

        def path_part(src, tgt):
            edge = graph.adj[src][tgt]
            return ConversionEdge(
                source=src, target=tgt, func=edge['func'], cost=edge['cost']
            )

        return map(path_part, shortest_pth, shortest_pth[1:])


def path_cost(path: ConversionPath) -> float:
    """Calculate the total cost of a path."""
    return sum(p.cost for p in path)


@contextmanager
def without_edges(g: nx.Graph, edges):
    edges = edges or []
    held = {}
    for a, b in edges:
        held[(a, b)] = g.adj[a][b]
        g.remove_edge(a, b)

    try:
        yield g
    finally:
        for (a, b), kwargs in held.items():
            g.add_edge(a, b, **kwargs)


if __name__ == "__main__":
    converter = ConversionGraph('test')

    from pydantic import BaseModel, EmailStr

    class From(BaseModel):
        name: str
        email: EmailStr

    @converter.register(BaseModel, dict)
    def test_convert(source: BaseModel, **kwargs):
        debug(source, kwargs)
        return source.dict()

    a = From(name='test', email='fty@gmail.com')
    res = converter(source=a, target=dict)
    debug(res)
