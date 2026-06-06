"""Agent data structure for Stratify simulation."""
from __future__ import annotations


class Agent:
    """Represents an individual in the simulation.

    Attributes:
        cls:   Current class/tier (non-negative integer).
        value: Accumulated wealth/resource value.
        ge:    Talent/gift value (positive).
        hard:  Effort value (positive).
        life:  Remaining lifespan.
        max_class: Historical highest class ever reached.
        age:   Ticks since birth.
        alive: Whether the agent is still alive.
    """

    def __init__(
        self,
        cls: int = 0,
        value: float = 0.0,
        ge: float = 1.0,
        hard: float = 1.0,
        life: float = 800.0,
    ) -> None:
        if cls < 0:
            raise ValueError(f"cls must be non-negative, got {cls}")
        if ge <= 0:
            raise ValueError(f"ge must be positive, got {ge}")
        if hard <= 0:
            raise ValueError(f"hard must be positive, got {hard}")
        if life <= 0:
            raise ValueError(f"life must be positive, got {life}")

        self._cls: int = cls
        self.value: float = value
        self.ge: float = ge
        self.hard: float = hard
        self.life: float = life
        self.max_class: int = cls
        self.age: int = 0
        self.alive: bool = True

    @property
    def cls(self) -> int:
        return self._cls

    @cls.setter
    def cls(self, new_cls: int) -> None:
        if new_cls < 0:
            raise ValueError(f"cls must be non-negative, got {new_cls}")
        self._cls = new_cls
        if new_cls > self.max_class:
            self.max_class = new_cls

    def __repr__(self) -> str:
        return (
            f"Agent(cls={self._cls}, value={self.value:.1f}, "
            f"ge={self.ge:.1f}, hard={self.hard:.1f}, "
            f"life={self.life:.1f}, alive={self.alive})"
        )