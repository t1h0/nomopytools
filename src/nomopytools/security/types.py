from typing import TypedDict, NamedTuple


class HasherPresets(TypedDict):
    length: int
    iterations: int
    lanes: int
    memory_cost: int


class PasswordProperties(NamedTuple):
    salt: bytes
    verifier: bytes
