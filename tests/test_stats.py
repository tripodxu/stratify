"""Tests for statistics snapshot module."""
import numpy as np
import pytest
from stratify.agent import Agent
from stratify.stats import (
    Snapshot,
    compute_snapshot,
    gini_coefficient,
    class_distribution,
    class_mean_values,
)


class TestGiniCoefficient:
    """基尼系数计算测试。"""

    def test_gini_perfect_equality(self):
        """所有人 value 相同 → 基尼系数 = 0。"""
        values = [100.0] * 100
        assert gini_coefficient(values) == pytest.approx(0.0, abs=1e-6)

    def test_gini_one_has_all(self):
        """一人拥有全部 → 基尼系数接近 (n-1)/n。"""
        values = [0.0] * 99 + [1000.0]
        g = gini_coefficient(values)
        assert g > 0.9

    def test_gini_partial_inequality(self):
        """部分不平等 → 基尼系数介于 0 和 1 之间。"""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        g = gini_coefficient(values)
        assert 0.0 < g < 1.0

    def test_gini_empty_returns_zero(self):
        """空列表返回 0。"""
        assert gini_coefficient([]) == pytest.approx(0.0)

    def test_gini_single_value(self):
        """单个值返回 0。"""
        assert gini_coefficient([42.0]) == pytest.approx(0.0)


class TestClassDistribution:
    """阶层人口分布测试。"""

    def test_even_distribution(self):
        """均匀分布时各层人数相等。"""
        agents = [Agent(cls=i % 3) for i in range(90)]
        dist = class_distribution(agents, num_classes=3)
        assert dist == [30, 30, 30]

    def test_uneven_distribution(self):
        """非均匀分布。"""
        agents = (
            [Agent(cls=0)] * 50
            + [Agent(cls=1)] * 30
            + [Agent(cls=2)] * 20
        )
        dist = class_distribution(agents, num_classes=3)
        assert dist == [50, 30, 20]

    def test_empty_agents(self):
        """无 agent 时返回全零。"""
        dist = class_distribution([], num_classes=3)
        assert dist == [0, 0, 0]


class TestClassMeanValues:
    """各阶层平均 Value 测试。"""

    def test_mean_values_by_class(self):
        """每层的平均 value 计算正确。"""
        a1 = Agent(cls=0, value=10.0, ge=1.0, hard=1.0, life=100.0)
        a2 = Agent(cls=0, value=20.0, ge=1.0, hard=1.0, life=100.0)
        a3 = Agent(cls=1, value=50.0, ge=1.0, hard=1.0, life=100.0)
        means = class_mean_values([a1, a2, a3], num_classes=2)
        assert means[0] == pytest.approx(15.0)
        assert means[1] == pytest.approx(50.0)

    def test_empty_class_mean_is_zero(self):
        """空阶层的平均值为 0。"""
        agents = [Agent(cls=0, value=10.0, ge=1.0, hard=1.0, life=100.0)]
        means = class_mean_values(agents, num_classes=3)
        assert means[0] == pytest.approx(10.0)
        assert means[1] == pytest.approx(0.0)
        assert means[2] == pytest.approx(0.0)


class TestComputeSnapshot:
    """完整快照计算测试。"""

    def test_snapshot_has_required_fields(self):
        """快照包含所有必要字段。"""
        agents = [Agent(cls=0, ge=5.0, hard=5.0, life=800.0) for _ in range(10)]
        snap = compute_snapshot(agents, tick=0, num_classes=2)
        assert isinstance(snap, Snapshot)
        assert snap.tick == 0
        assert snap.population == 10
        assert snap.alive_count == 10
        assert len(snap.class_counts) == 2
        assert len(snap.class_mean_value) == 2
        assert snap.gini >= 0.0
        assert snap.total_value >= 0.0

    def test_snapshot_counts_alive_correctly(self):
        """快照正确统计存活人数。"""
        agents = [Agent(cls=0, ge=5.0, hard=5.0, life=800.0) for _ in range(10)]
        agents[3].alive = False
        agents[7].alive = False
        snap = compute_snapshot(agents, tick=0, num_classes=1)
        assert snap.alive_count == 8
        assert snap.population == 10

    def test_snapshot_gini_after_accumulation(self):
        """积累后不同阶层的 value 不同，基尼系数 > 0。"""
        from stratify.world import World
        w = World(num_agents=100, num_classes=5, seed=42)
        for _ in range(50):
            w.tick()
        snap = compute_snapshot(w.agents, tick=w.tick_count, num_classes=5)
        assert snap.gini > 0.0