import itertools
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Tuple,
    Type,
    TypeVar,
    cast,
)
from warnings import warn
from graphene.types.objecttype import ObjectType, ObjectTypeOptions
import graphene
from graphene.types.scalars import Scalar

import networkx as nx
from devtools import debug

# from networkx.exception import NetworkXNoPath

from src.utils.iter import IterProxy
from src.utils.graphs import without_edges
from src.utils.typing import ConverterFunc


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


_S = TypeVar('_S')  # , bound=type
_T = TypeVar('_T')  # , bound=type


@dataclass
class ConversionEdge(Generic[_S, _T]):
    source: Type[_S]
    target: Type[_T]
    func: ConverterFunc[_S, _T]
    cost: float


_virtual_superclasses = (Iterator,)


class ConversionGraph:
    def __init__(self, name: str):
        self.name = name
        # self._graph = nx.MultiDiGraph(name=name)
        self.graph = nx.DiGraph(name=name)

    def __call__(self, source, target, **kwargs):
        return self._transform(source=source, target=target, **kwargs)

    def register(
        self, source: Type[_S], target: Type[_T], cost=1.0
    ) -> Callable[[ConverterFunc[_S, _T]], ConverterFunc[_S, _T]]:
        """Decorator that registers a conversion fn between 2 types

        Args:
            source: The type of the object being converted
            target: The type to convert to
        """
        # https://github.com/blaze/odo/blob/master/odo/core.py#L70

        def _wrapper(func: ConverterFunc):
            self.graph.add_edge(source, target, cost=cost, func=func)
            return func

        return _wrapper

    def path(
        self,
        source: Type[_S],
        target: Type[_T],
        excluded_edges: Iterable[Tuple[type, type]] = None,
    ) -> Iterator[ConversionEdge[_S, _T]]:
        """Path of functions between two types"""
        debug(source, target, self.graph.edges)

        source = self._get_type_node(source)  # type: ignore
        target = self._get_type_node(target)  # type: ignore

        with without_edges(self.graph, excluded_edges) as g:
            shortest_pth = cast(
                list, nx.shortest_path(g, source=source, target=target, weight='cost')
            )
            debug(shortest_pth)
            return map(self._get_conversion_edge, shortest_pth, shortest_pth[1:])

    def _transform(
        self,
        source: _S,
        target: Type[_T],
        excluded_edges: Iterable[Tuple[type, type]] = None,
        raise_on_errors: bool = False,
        **kwargs,
    ) -> _T:
        """Transform source to target type using graph of transformations"""
        # take a copy so we can mutate without affecting the input
        excluded_edges = (
            set(excluded_edges).copy() if excluded_edges is not None else set()
        )

        path = self.path(
            source=type(source), target=target, excluded_edges=excluded_edges
        )
        path_proxy: IterProxy[ConversionEdge[_S, _T]] = IterProxy(path)

        step = source
        for edge in path_proxy:
            try:
                step = edge.func(
                    step,  # type: ignore
                    target=target,
                    excluded_edges=excluded_edges,
                    **kwargs,
                )
            except NotImplementedError as e:
                if raise_on_errors:
                    raise
                warn(FailedConversionWarning(edge.source, edge.target, e))
                # exclude the broken edge
                excluded_edges.add((edge.source, edge.target))

                # compute the path from `source` to `target` excluding
                # the edge that broke
                path = list(
                    self.path(type(source), target, excluded_edges=excluded_edges)
                )
                # fresh_path_cost = path_cost(fresh_path)
                # ? Should we keep this?
                # # compute the path from the current `source` type
                # # to the `target`
                # try:
                #     greedy_path = list(
                #         self.path(
                #             edge.target, target, excluded_edges=excluded_edges
                #         )
                #     )
                #     except NetworkXNoPath:
                #         greedy_path_cost = float("inf")
                #     else:
                #         greedy_path_cost = path_cost(greedy_path)

                #     if fresh_path_cost < greedy_path_cost:
                #         # it is faster to start over from `source` with a new path
                #         step = source
                #         path = fresh_path
                #     else:
                #         # it is faster to work around our broken edge from our
                #         # current location
                #         path = greedy_path
                path_proxy.it = path

        return cast(_T, step)

    def _get_conversion_edge(
        self, source: Type[_S], target: Type[_T]
    ) -> ConversionEdge[_S, _T]:
        edge = self.graph.adj[source][target]
        return ConversionEdge(
            source=source, target=target, func=edge['func'], cost=edge['cost']
        )

    def _get_type_node(self, cls: type):
        """Traverses a type's MRO and returns the first class present in the Graph"""
        if not isinstance(cls, type):
            cls = type(cls)

        for clss in itertools.chain(cls.mro(), _virtual_superclasses):
            if clss in self.graph:
                node = clss
                break
        else:
            raise ValueError(
                f'Neither {cls=} nor any of it\'s base classes are '
                f'registered in the graph: {cls.__mro__=}, {self.graph.edges=}'
            )

        return node


CONVERTER_FUNCS = {
    (src, tgt): src.serialize
    for src, tgt in [
        (graphene.String, str),
        (graphene.Int, int),
    ]
}


def register_converter_funcs(
    graph: ConversionGraph,
    converter_funcs: Dict[Tuple[Type[_S], Type[_T]], ConverterFunc[_S, _T]],
):
    for (source, target), func in converter_funcs.items():
        graph.register(source, target)(func)


if __name__ == "__main__":
    converter = ConversionGraph('test')
    register_converter_funcs(converter, CONVERTER_FUNCS)

    from pydantic import BaseModel, EmailStr

    class From(BaseModel):
        name: str
        email: EmailStr

    @converter.register(BaseModel, dict)
    def basemodel_to_dict(source: BaseModel, **kwargs) -> dict:
        debug(source, kwargs)
        return source.dict()

    a = From(name='test', email='fty@gmail.com')
    res = converter(source=a, target=dict)
    debug(res)

    class FromObjectType(ObjectType):
        name = graphene.String()
        email = graphene.String()

    @converter.register(ObjectType, dict)
    def objecttype_to_dict(source: ObjectType, **kwargs):
        debug(source, kwargs)

        res = {}
        for field_name in cast(ObjectTypeOptions, source._meta).fields:
            field_type = getattr(type(source), field_name)

            if isinstance(field_type, Scalar):
                field_value = getattr(source, field_name)
                res[field_name] = field_value

            # else:
            #     res[field_name] = converter(source=field_type, target=)??

        return res

    @converter.register(dict, ObjectType)
    def dict_to_objecttype(source: dict, target: Type[ObjectType], **kwargs):
        debug(source, kwargs, target)
        return target(**source)

    @converter.register(dict, BaseModel)
    def dict_to_basemodel(source: dict, target: Type[BaseModel], **kwargs):
        debug(source, target)
        return target(**source)

    objtype = FromObjectType('asd', 'asd')
    res = converter(source=objtype, target=dict)
    debug(res)

    res = converter(source=a, target=FromObjectType)
    debug(res)
