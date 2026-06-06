"""Integration tests for Phase 1 experiment (pure accumulation)."""
import numpy as np
import pytest
from stratify.world import World
from stratify.stats import compute_snapshot


class TestPhase1Integration:
    """阶段一集成实验：纯积累，无竞争，无寿命。"""

    @pytest.fixture
    def phase1_world(self) -> World:
        return World.phase1(seed=42)

    def test_total_value_increases_over_time(self, phase1_world: World):
        """总 Value 随时间单调递增。"""
        w = phase1_world
        snap0 = compute_snapshot(w.agents, 0, w.num_classes)
        for _ in range(100):
            w.tick()
        snap100 = compute_snapshot(w.agents, w.tick_count, w.num_classes)
        assert snap100.total_value > snap0.total_value

    def test_gini_increases_over_time(self, phase1_world: World):
        """基尼系数随时间上升（阶层差距拉大）。"""
        w = phase1_world
        snap0 = compute_snapshot(w.agents, 0, w.num_classes)
        for _ in range(200):
            w.tick()
        snap200 = compute_snapshot(w.agents, w.tick_count, w.num_classes)
        assert snap200.gini > snap0.gini

    def test_higher_class_has_higher_mean_value(self, phase1_world: World):
        """高阶层平均 Value 高于低阶层。"""
        w = phase1_world
        for _ in range(500):
            w.tick()
        snap = compute_snapshot(w.agents, w.tick_count, w.num_classes)
        for i in range(w.num_classes - 1):
            assert snap.class_mean_value[i + 1] > snap.class_mean_value[i], (
                f"Class {i+1} mean ({snap.class_mean_value[i+1]:.1f}) "
                f"should exceed class {i} mean ({snap.class_mean_value[i]:.1f})"
            )

    def test_class_premium_with_same_attributes(self, phase1_world: World):
        """相同 ge/hard 的人，阶层越高积累越多（阶层溢价）。"""
        w = World(
            num_agents=2, num_classes=2,
            env_coefficients=[1.0, 6.0],
            seed=0,
        )
        # 强制设定完全相同的 ge/hard
        for a in w.agents:
            a.ge = 5.0
            a.hard = 5.0
        w.agents[0].cls = 0
        w.agents[1].cls = 1
        for _ in range(100):
            w.tick()
        ratio = w.agents[1].value / w.agents[0].value
        assert ratio == pytest.approx(6.0, rel=0.01), (
            f"Expected 6x class premium, got {ratio:.2f}x"
        )

    def test_value_spread_widens_over_time(self, phase1_world: World):
        """Value 标准差随时间增大。"""
        w = phase1_world
        std_0 = np.std([a.value for a in w.agents])
        for _ in range(300):
            w.tick()
        std_300 = np.std([a.value for a in w.agents])
        assert std_300 > std_0

    def test_all_agents_stay_alive_phase1(self, phase1_world: World):
        """阶段一无死亡机制，所有人都应存活。"""
        w = phase1_world
        for _ in range(100):
            w.tick()
        assert all(a.alive for a in w.agents)

    def test_snapshot_history_tracking(self, phase1_world: World):
        """可以追踪多 tick 的快照历史。"""
        w = phase1_world
        history = []
        for t in range(50):
            w.tick()
            snap = compute_snapshot(w.agents, w.tick_count, w.num_classes)
            history.append(snap)
        assert len(history) == 50
        assert history[0].tick == 1
        assert history[-1].tick == 50
        # Value 应该持续增长
        for i in range(1, len(history)):
            assert history[i].total_value > history[i - 1].total_value

    def test_phase1_deterministic_with_seed(self):
        """固定 seed 结果可复现。"""
        w1 = World.phase1(seed=123)
        w2 = World.phase1(seed=123)
        for _ in range(100):
            w1.tick()
            w2.tick()
        for a1, a2 in zip(w1.agents, w2.agents):
            assert a1.value == pytest.approx(a2.value)
            assert a1.ge == pytest.approx(a2.ge)