"""
Copy-on-Write Proxies for Mutable State Isolation.

This module provides `CoWList` and `CoWDict` wrappers that ensure modifications
to mutable objects inside the agent's `_ctx` trigger a copy-on-write update,
maintaining branch isolation.

Key Features:
- Recursive wrapping: Nested mutable objects are also wrapped on access
- Path copying: Modifying a nested leaf triggers CoW up the entire path

Usage:
    # The compiler automatically wraps mutable values when reading from _ctx.
    # Users should not need to interact with these classes directly.
"""

from collections.abc import Callable, Iterable, Iterator, MutableMapping, MutableSequence
from typing import Any, TypeVar

T = TypeVar("T")
KT = TypeVar("KT")
VT = TypeVar("VT")


class CoWList(MutableSequence[T]):
    """
    A Copy-on-Write list wrapper with recursive child wrapping.
    
    When a mutating operation is performed, the list is copied, mutated,
    and the callback is invoked with the new list.
    
    When accessing a mutable child (via __getitem__), the child is wrapped
    with a callback that triggers copy-on-write on the parent.
    """
    
    __slots__ = ("_data", "_callback", "_copied")
    
    def __init__(self, data: list[T], callback: Callable[[list[T]], None]):
        """
        Args:
            data: The underlying list to wrap.
            callback: Called with the new list after any mutation.
                      Signature: callback(new_list) -> None
        """
        self._data = data
        self._callback = callback
        self._copied = False
    
    def _ensure_copy(self) -> None:
        """Ensure we have our own copy before mutating."""
        if not self._copied:
            self._data = list(self._data)
            self._copied = True
    
    def _notify(self) -> None:
        """Notify the parent context of the mutation."""
        self._callback(self._data)
    
    # === Read operations with recursive wrapping ===
    
    def __getitem__(self, index: int | slice) -> T | list[T]:
        val = self._data[index]
        
        # For slices, return a plain list (no wrapping needed for read-only slices)
        if isinstance(index, slice):
            return val
        
        # Recursively wrap mutable children so mutations propagate up
        if isinstance(val, list) and not isinstance(val, CoWList):
            def child_callback(new_child: list) -> None:
                # When child is mutated, update ourselves
                self[index] = new_child  # This triggers our _ensure_copy and _notify
            return CoWList(val, child_callback)
        
        if isinstance(val, dict) and not isinstance(val, CoWDict):
            def child_callback(new_child: dict) -> None:
                self[index] = new_child
            return CoWDict(val, child_callback)
        
        return val
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[T]:
        # Wrap mutable items during iteration
        for i in range(len(self._data)):
            yield self[i]  # Uses __getitem__ which wraps if needed
    
    def __contains__(self, item: object) -> bool:
        return item in self._data
    
    def __repr__(self) -> str:
        return f"CoWList({self._data!r})"
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, CoWList):
            return self._data == other._data
        return self._data == other
    
    def index(self, value: T, start: int = 0, stop: int | None = None) -> int:
        if stop is None:
            return self._data.index(value, start)
        return self._data.index(value, start, stop)
    
    def count(self, value: T) -> int:
        return self._data.count(value)
    
    # === Write operations (copy first) ===
    
    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        self._ensure_copy()
        # Unwrap CoW proxies before storing
        if isinstance(value, (CoWList, CoWDict)):
            value = value.unwrap()
        self._data[index] = value  # type: ignore
        self._notify()
    
    def __delitem__(self, index: int | slice) -> None:
        self._ensure_copy()
        del self._data[index]
        self._notify()
    
    def __iadd__(self, other: Iterable[T]) -> "CoWList[T]":
        """Support += operator."""
        self._ensure_copy()
        self._data.extend(other)
        self._notify()
        return self
    
    def __add__(self, other: Iterable[T]) -> list[T]:
        """Support + operator: returns a new plain list."""
        result = list(self._data)
        result.extend(other)
        return result
    
    def __radd__(self, other: Iterable[T]) -> list[T]:
        """Support reversed + operator."""
        result = list(other)
        result.extend(self._data)
        return result
    
    def insert(self, index: int, value: T) -> None:
        self._ensure_copy()
        self._data.insert(index, value)
        self._notify()
    
    def append(self, value: T) -> None:
        self._ensure_copy()
        self._data.append(value)
        self._notify()
    
    def extend(self, values: Iterable[T]) -> None:
        self._ensure_copy()
        self._data.extend(values)
        self._notify()
    
    def pop(self, index: int = -1) -> T:
        self._ensure_copy()
        result = self._data.pop(index)
        self._notify()
        return result
    
    def remove(self, value: T) -> None:
        self._ensure_copy()
        self._data.remove(value)
        self._notify()
    
    def clear(self) -> None:
        self._ensure_copy()
        self._data.clear()
        self._notify()
    
    def reverse(self) -> None:
        self._ensure_copy()
        self._data.reverse()
        self._notify()
    
    def sort(self, *, key: Callable[[T], Any] | None = None, reverse: bool = False) -> None:
        self._ensure_copy()
        self._data.sort(key=key, reverse=reverse)
        self._notify()
    
    def copy(self) -> list[T]:
        """Return a plain list copy (not wrapped)."""
        return list(self._data)
    
    def unwrap(self) -> list[T]:
        """Return the underlying list (for serialization)."""
        return self._data


class CoWDict(MutableMapping[KT, VT]):
    """
    A Copy-on-Write dict wrapper with recursive child wrapping.
    
    When a mutating operation is performed, the dict is copied, mutated,
    and the callback is invoked with the new dict.
    
    When accessing a mutable child (via __getitem__), the child is wrapped
    with a callback that triggers copy-on-write on the parent.
    """
    
    __slots__ = ("_data", "_callback", "_copied")
    
    def __init__(self, data: dict[KT, VT], callback: Callable[[dict[KT, VT]], None]):
        """
        Args:
            data: The underlying dict to wrap.
            callback: Called with the new dict after any mutation.
        """
        self._data = data
        self._callback = callback
        self._copied = False
    
    def _ensure_copy(self) -> None:
        """Ensure we have our own copy before mutating."""
        if not self._copied:
            self._data = dict(self._data)
            self._copied = True
    
    def _notify(self) -> None:
        """Notify the parent context of the mutation."""
        self._callback(self._data)
    
    # === Read operations with recursive wrapping ===
    
    def __getitem__(self, key: KT) -> VT:
        val = self._data[key]
        
        # Recursively wrap mutable children so mutations propagate up
        if isinstance(val, list) and not isinstance(val, CoWList):
            def child_callback(new_child: list) -> None:
                self[key] = new_child  # Triggers our _ensure_copy and _notify
            return CoWList(val, child_callback)  # type: ignore
        
        if isinstance(val, dict) and not isinstance(val, CoWDict):
            def child_callback(new_child: dict) -> None:
                self[key] = new_child
            return CoWDict(val, child_callback)  # type: ignore
        
        return val
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[KT]:
        return iter(self._data)
    
    def __contains__(self, key: object) -> bool:
        return key in self._data
    
    def __repr__(self) -> str:
        return f"CoWDict({self._data!r})"
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, CoWDict):
            return self._data == other._data
        return self._data == other
    
    def get(self, key: KT, default: VT | None = None) -> VT | None:
        if key in self._data:
            return self[key]  # Use __getitem__ to get wrapped value
        return default
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        # Wrap mutable values
        for k in self._data:
            yield self[k]
    
    def items(self):
        # Wrap mutable values
        for k in self._data:
            yield (k, self[k])
    
    # === Write operations ===
    
    def __setitem__(self, key: KT, value: VT) -> None:
        self._ensure_copy()
        # Unwrap CoW proxies before storing
        if isinstance(value, (CoWList, CoWDict)):
            value = value.unwrap()  # type: ignore
        self._data[key] = value
        self._notify()
    
    def __delitem__(self, key: KT) -> None:
        self._ensure_copy()
        del self._data[key]
        self._notify()
    
    def pop(self, key: KT, *default: VT) -> VT:
        self._ensure_copy()
        result = self._data.pop(key, *default)
        self._notify()
        return result
    
    def popitem(self) -> tuple[KT, VT]:
        self._ensure_copy()
        result = self._data.popitem()
        self._notify()
        return result
    
    def clear(self) -> None:
        self._ensure_copy()
        self._data.clear()
        self._notify()
    
    def update(self, other: dict[KT, VT] | Iterable[tuple[KT, VT]] = (), **kwargs: VT) -> None:
        self._ensure_copy()
        self._data.update(other, **kwargs)
        self._notify()
    
    def setdefault(self, key: KT, default: VT | None = None) -> VT | None:
        if key not in self._data:
            self._ensure_copy()
            self._data[key] = default  # type: ignore
            self._notify()
        return self[key]  # Use __getitem__ for wrapped access
    
    def copy(self) -> dict[KT, VT]:
        """Return a plain dict copy (not wrapped)."""
        return dict(self._data)
    
    def unwrap(self) -> dict[KT, VT]:
        """Return the underlying dict (for serialization)."""
        return self._data


def wrap_mutable(value: Any, key: str, ctx_setter: Callable[[str, Any], Any]) -> Any:
    """
    Wrap a mutable value in a CoW proxy if needed.
    
    Args:
        value: The value to potentially wrap.
        key: The key in _ctx where this value is stored.
        ctx_setter: A callable that sets _ctx[key] = new_value.
                   Signature: ctx_setter(key, new_value) -> new_ctx
    
    Returns:
        The wrapped value (if mutable) or the original value.
    """
    if isinstance(value, list) and not isinstance(value, CoWList):
        def callback(new_list: list) -> None:
            ctx_setter(key, new_list)
        return CoWList(value, callback)
    
    if isinstance(value, dict) and not isinstance(value, CoWDict):
        def callback(new_dict: dict) -> None:
            ctx_setter(key, new_dict)
        return CoWDict(value, callback)
    
    return value
