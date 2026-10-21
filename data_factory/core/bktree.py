"""A BK-tree for near-neighbour search in Hamming space.

Perceptual dedup needs "find everything within N bits of this hash". Doing that
by comparing every image to every kept image is O(n^2): on the 15k-image
door dataset that is ~112 million comparisons, which is why the original
script was only ever run on small folders.

A BK-tree exploits the triangle inequality -- Hamming distance is a metric, so
a child at distance k from a node can only contain matches within [d-r, d+r] of
the query. Whole subtrees get skipped, and lookup drops to roughly logarithmic
for the small radii (0-10 bits) that dedup actually uses.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any


class _Node:
    __slots__ = ("children", "item")

    def __init__(self, item: Any) -> None:
        self.item = item
        self.children: dict[int, _Node] = {}


class BKTree:
    """Metric tree supporting ``add`` and radius search.

    ``distance`` must be a true metric (symmetric, triangle inequality) --
    Hamming distance between equal-length hashes qualifies.
    """

    def __init__(self, distance: Callable[[Any, Any], int]) -> None:
        self._distance = distance
        self._root: _Node | None = None
        self._size = 0

    def add(self, item: Any) -> None:
        self._size += 1
        if self._root is None:
            self._root = _Node(item)
            return
        node = self._root
        while True:
            gap = self._distance(item, node.item)
            child = node.children.get(gap)
            if child is None:
                node.children[gap] = _Node(item)
                return
            node = child

    def find(self, query: Any, radius: int) -> list[tuple[int, Any]]:
        """Every stored item within ``radius``, as ``(distance, item)`` pairs."""
        if self._root is None:
            return []
        matches: list[tuple[int, Any]] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            gap = self._distance(query, node.item)
            if gap <= radius:
                matches.append((gap, node.item))
            # Triangle inequality: only these children can hold a match.
            low, high = gap - radius, gap + radius
            for edge, child in node.children.items():
                if low <= edge <= high:
                    stack.append(child)
        matches.sort(key=lambda pair: pair[0])
        return matches

    def nearest(self, query: Any, radius: int) -> tuple[int, Any] | None:
        found = self.find(query, radius)
        return found[0] if found else None

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        if self._root is None:
            return iter(())
        items: list[Any] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            items.append(node.item)
            stack.extend(node.children.values())
        return iter(items)
