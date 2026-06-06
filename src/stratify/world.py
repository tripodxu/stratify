"""World simulation engine for Stratify."""
from __future__ import annotations

import numpy as np

from stratify.agent import Agent
from stratify.accumulation import accumulation_speed


class World:
    """The simulation world that manages agents and tick cycles.

    Phase 1 mode: pure accumulation only (no competition, no lifespan decay).
    """

    def __init__(
        self,
        num_agents: int = 1000,
        num_classes: int = 5,
        env_coefficients: list[float] | None = None,
        ge_mean: float = 5.0,
        ge_std: float = 2.0,
        ge_min: float = 1.0,
        ge_max: float = 10.0,
        hard_mean: float = 5.0,
        hard_std: float = 2.0,
        hard_min: float = 1.0,
        hard_max: float = 10.0,
        seed: int | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.env_coefficients = env_coefficients or [
            float(i + 1) for i in range(num_classes)
        ]
        self.tick_count: int = 0
        self._rng = np.random.default_rng(seed)

        # Generate agents with random ge/hard, evenly distributed across classes
        agents_per_class = num_agents // num_classes
        remainder = num_agents % num_classes

        self.agents: list[Agent] = []
        for cls_idx in range(num_classes):
            count = agents_per_class + (1 if cls_idx < remainder else 0)
            ge_vals = self._truncated_normal(
                count, ge_mean, ge_std, ge_min, ge_max
            )
            hard_vals = self._truncated_normal(
                count, hard_mean, hard_std, hard_min, hard_max
            )
            for g, h in zip(ge_vals, hard_vals):
                self.agents.append(
                    Agent(cls=cls_idx, value=0.0, ge=float(g), hard=float(h), life=800.0)
                )

    def _truncated_normal(
        self, n: int, mean: float, std: float, lo: float, hi: float
    ) -> np.ndarray:
        """Generate n samples from a truncated normal distribution."""
        samples = self._rng.normal(mean, std, size=n)
        return np.clip(samples, lo, hi)

    def tick(self) -> None:
        """Advance the simulation by one tick.

        Phase 1: accumulate value for each alive agent.
        """
        for agent in self.agents:
            if not agent.alive:
                continue
            speed = accumulation_speed(
                env=self.env_coefficients[agent.cls],
                ge=agent.ge,
                hard=agent.hard,
            )
            agent.value += speed
            agent.age += 1
        self.tick_count += 1

    # ----- Factory methods for preset experiments -----

    @classmethod
    def phase1(cls, seed: int | None = None) -> World:
        """Create a Phase 1 world: 1000 agents, 5 classes, pure accumulation.

        Parameters match plan.md Phase 1:
            - 1000 agents, evenly distributed
            - 5 classes, env = [1.0, 1.5, 2.5, 4.0, 6.0]
            - ge ~ N(5, 2), truncated [1, 10]
            - hard ~ N(5, 2), truncated [1, 10]
        """
        return cls(
            num_agents=1000,
            num_classes=5,
            env_coefficients=[1.0, 1.5, 2.5, 4.0, 6.0],
            ge_mean=5.0,
            ge_std=2.0,
            ge_min=1.0,
            ge_max=10.0,
            hard_mean=5.0,
            hard_std=2.0,
            hard_min=1.0,
            hard_max=10.0,
            seed=seed,
        )