from typing import Generic, Iterable, TypeVar


_T = TypeVar('_T')


class IterProxy(Generic[_T]):
    """An proxy to another iterator to support swapping the underlying stream
    mid-iteration.
    Parameters
    ----------
    it : iterable
        The iterable to proxy.
    Attributes
    ----------
    it : iterable
        The iterable being proxied. This can be reassigned to change the
        underlying stream.
    """

    def __init__(self, it: Iterable[_T]):
        self._it = iter(it)

    def __next__(self):
        return next(self.it)

    def __iter__(self):
        return self

    @property
    def it(self):
        return self._it

    @it.setter
    def it(self, value):
        self._it = iter(value)
