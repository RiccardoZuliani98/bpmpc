"""Symbolic variable descriptor for parametric MPC problems."""

from __future__ import annotations

from typing import Tuple


class Variable:
    """A named symbolic variable with a fixed shape.

    A *descriptor* — carries no numerical data.  It tells the MPC
    builder what arrays the user's constraint/cost functions expect
    so that the builder can wire everything together.

    Parameters
    ----------
    name  : Human-readable identifier (unique within a problem).
    shape : Shape of the concrete JAX array this variable represents.
            Scalars are expressed as ``(1,)``.
    """

    __slots__ = ("name", "shape")

    name:  str
    shape: Tuple[int, ...]

    def __init__(self, name: str, shape: Tuple[int, ...]) -> None:
        if not isinstance(shape, tuple) or not all(
            isinstance(s, int) and s > 0 for s in shape
        ):
            raise ValueError(
                f"Variable '{name}': shape must be a tuple of positive "
                f"ints, got {shape!r}"
            )
        self.name  = name
        self.shape = shape

    def __repr__(self) -> str:
        return f"Variable(name={self.name!r}, shape={self.shape})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Variable):
            return NotImplemented
        return self.name == other.name and self.shape == other.shape

    def __hash__(self) -> int:
        return hash((self.name, self.shape))
