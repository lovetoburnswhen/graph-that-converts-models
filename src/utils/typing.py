

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph import ConversionEdge


ConversionPath = List["ConversionEdge"]
