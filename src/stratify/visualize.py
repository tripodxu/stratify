"""Visualization utilities for Stratify simulation."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stratify.stats import Snapshot


def plot_class_distribution_stack(
    history: list[Snapshot],
    num_classes: int,
    save_path: str | Path | None = None,
) -> None:
    """Plot stacked area chart of class population over time."""
    ticks = [s.tick for s in history]
    data = np.array([s.class_counts for s in history])  # shape (T, num_classes)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(
        ticks,
        *[data[:, i] for i in range(num_classes)],
        labels=[f"Class {i}" for i in range(num_classes)],
        alpha=0.8,
    )
    ax.set_xlabel("Tick")
    ax.set_ylabel("Population")
    ax.set_title("Class Distribution Over Time")
    ax.legend(loc="upper right")
    ax.set_xlim(ticks[0], ticks[-1])
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_gini_curve(
    history: list[Snapshot],
    save_path: str | Path | None = None,
) -> None:
    """Plot Gini coefficient over time."""
    ticks = [s.tick for s in history]
    ginis = [s.gini for s in history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ticks, ginis, color="#e74c3c", linewidth=1.5)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Value Inequality (Gini) Over Time")
    ax.set_ylim(0, max(ginis) * 1.1 + 0.01)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_class_mean_value_curves(
    history: list[Snapshot],
    num_classes: int,
    save_path: str | Path | None = None,
) -> None:
    """Plot mean value per class over time."""
    ticks = [s.tick for s in history]
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, num_classes))
    for i in range(num_classes):
        means = [s.class_mean_value[i] for s in history]
        ax.plot(ticks, means, label=f"Class {i}", color=colors[i], linewidth=1.5)

    ax.set_xlabel("Tick")
    ax.set_ylabel("Mean Value")
    ax.set_title("Mean Value by Class Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_value_histogram(
    history: list[Snapshot],
    snapshots_to_plot: list[int] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Placeholder for value histogram at selected ticks.

    This function requires the full agent list which Snapshot doesn't carry,
    so it's a stub for now.
    """
    pass