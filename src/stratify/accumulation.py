"""Value accumulation formulas for Stratify simulation."""
from __future__ import annotations


def accumulation_speed(env: float, ge: float, hard: float) -> float:
    """Calculate the value accumulation speed per tick.

    Formula: speed = env * (ge + hard) / 2

    Args:
        env:  Class/tier coefficient (higher tier → higher coefficient).
        ge:   Talent/gift value (positive).
        hard: Effort value (positive).

    Returns:
        The amount of value gained per tick.
    """
    return env * (ge + hard) / 2


def accumulate_value(current_value: float, speed: float) -> float:
    """Add one tick's worth of accumulation to the current value.

    Args:
        current_value: The agent's current value before this tick.
        speed:         The accumulation speed for this tick.

    Returns:
        The new value after accumulation.
    """
    return current_value + speed