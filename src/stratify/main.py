"""Stratify simulation - main entry point.

Usage:
    python -m stratify.main           # Run Phase 1 experiment
    python -m stratify.main --ticks 2000  # Custom tick count
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stratify.world import World
from stratify.stats import compute_snapshot
from stratify.visualize import (
    plot_class_distribution_stack,
    plot_gini_curve,
    plot_class_mean_value_curves,
)


def run_phase1(ticks: int = 1000, output_dir: str = "output") -> None:
    """Run the Phase 1 experiment and generate visualizations."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== Stratify Phase 1: Pure Accumulation ===")
    print(f"Agents: 1000 | Classes: 5 | Ticks: {ticks}")
    print(f"Env: [1.0, 1.5, 2.5, 4.0, 6.0]")
    print(f"ge ~ N(5, 2), hard ~ N(5, 2)")
    print()

    w = World.phase1(seed=42)
    history: list = []

    # Collect initial snapshot
    history.append(compute_snapshot(w.agents, 0, w.num_classes))

    for t in range(1, ticks + 1):
        w.tick()
        if t % 10 == 0 or t == ticks:
            snap = compute_snapshot(w.agents, w.tick_count, w.num_classes)
            history.append(snap)

        if t % 200 == 0 or t == ticks:
            snap = history[-1]
            print(
                f"  tick {t:>5d} | "
                f"total_value={snap.total_value:>12.1f} | "
                f"gini={snap.gini:.4f} | "
                f"class_means=[{', '.join(f'{m:.1f}' for m in snap.class_mean_value)}]"
            )

    # Generate plots
    print()
    print("Generating plots...")

    plot_class_distribution_stack(
        history, w.num_classes, save_path=out / "class_distribution.png"
    )
    print(f"  -> {out / 'class_distribution.png'}")

    plot_gini_curve(history, save_path=out / "gini_curve.png")
    print(f"  -> {out / 'gini_curve.png'}")

    plot_class_mean_value_curves(
        history, w.num_classes, save_path=out / "class_mean_values.png"
    )
    print(f"  -> {out / 'class_mean_values.png'}")

    print()
    print("Done. Phase 1 experiment complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratify Simulation")
    parser.add_argument("--ticks", type=int, default=1000, help="Number of ticks to run")
    parser.add_argument("--output", type=str, default="output", help="Output directory for plots")
    args = parser.parse_args()

    run_phase1(ticks=args.ticks, output_dir=args.output)


if __name__ == "__main__":
    main()