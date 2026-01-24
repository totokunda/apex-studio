from __future__ import annotations

from typing import Optional, Sequence, TypeVar

T = TypeVar("T")


def unwrap_single_item_list(value: Optional[Sequence[T]]) -> Optional[T]:
    """
    Unwrap a single-item list/tuple and clear list references early.

    If `value` is a list or tuple containing a single item, return that item. If it's
    a list, it will be cleared in-place to drop references eagerly.
    """
    if value is None:
        return None

    if isinstance(value, list):
        if not value:
            return None
        item = value[0]
        value.clear()
        return item

    if isinstance(value, tuple):
        return value[0] if value else None

    return value
