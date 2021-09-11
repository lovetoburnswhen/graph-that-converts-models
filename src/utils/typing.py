

from typing import Any, Generic, List, TYPE_CHECKING, Optional, Protocol, TypeVar

if TYPE_CHECKING:
    from src.graph import ConversionEdge


ConversionPath = List["ConversionEdge"]

_S = TypeVar('_S', contravariant=True)
_T = TypeVar('_T', covariant=True)


class ConverterFunc(Protocol, Generic[_S, _T]):
    # def __call__(self, source: _S, *, raise_on_errors: Optional[bool], **kwargs) -> _T:
    def __call__(self, source: _S, **kwargs: Any) -> _T:
        ...
