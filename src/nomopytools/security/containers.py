from typing import NamedTuple


class HasherPresets(NamedTuple):
    length: int
    iterations: int
    lanes: int
    memory_cost: int
