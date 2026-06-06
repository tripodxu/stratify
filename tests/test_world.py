"""Tests for World simulation tick cycle."""
import pytest
from stratify.agent import Agent
from stratify.world import World


class TestWorldCreation:
    """World 创建与基本配置测试。"""

    def test_create_world_with_default_params(self):
        """创建默认 World。"""
        w = World(num_agents=100, num_classes=5)
        assert w.tick_count == 0
        assert len(w.agents) == 100
        assert w.num_classes == 5

    def test_create_world_with_custom_env(self):
        """自定义 env 系数。"""
        env = [1.0, 2.0, 3.0]
        w = World(num_agents=50, num_classes=3, env_coefficients=env)
        assert w.env_coefficients == [1.0, 2.0, 3.0]

    def test_agents_distributed_evenly_by_default(self):
        """默认均匀分配到各阶层。"""
        w = World(num_agents=100, num_classes=5)
        class_counts = [0] * 5
        for a in w.agents:
            class_counts[a.cls] += 1
        assert all(c == 20 for c in class_counts)

    def test_agents_have_valid_attributes(self):
        """所有 agent 的属性在合法范围内。"""
        w = World(num_agents=50, num_classes=3)
        for a in w.agents:
            assert a.ge > 0
            assert a.hard > 0
            assert a.life > 0
            assert 0 <= a.cls < 3
            assert a.alive is True


class TestWorldTick:
    """World tick 循环测试。"""

    def test_tick_increments_counter(self):
        """每次 tick 计数器 +1。"""
        w = World(num_agents=10, num_classes=3)
        assert w.tick_count == 0
        w.tick()
        assert w.tick_count == 1
        w.tick()
        assert w.tick_count == 2

    def test_tick_accumulates_value(self):
        """tick 后 agent 的 value 应增加。"""
        w = World(num_agents=10, num_classes=3, env_coefficients=[1.0, 2.0, 3.0])
        initial_values = [a.value for a in w.agents]
        w.tick()
        for i, a in enumerate(w.agents):
            assert a.value > initial_values[i], f"Agent {i} value should increase"

    def test_higher_class_accumulates_faster(self):
        """高阶层 agent 积累更快（同 ge/hard 条件下）。"""
        env = [1.0, 5.0]
        w = World(num_agents=2, num_classes=2, env_coefficients=env)
        # 人为设定相同的 ge 和 hard
        w.agents[0].cls = 0
        w.agents[0].ge = 5.0
        w.agents[0].hard = 5.0
        w.agents[0].value = 0.0
        w.agents[1].cls = 1
        w.agents[1].ge = 5.0
        w.agents[1].hard = 5.0
        w.agents[1].value = 0.0
        w.tick()
        assert w.agents[1].value > w.agents[0].value

    def test_tick_ages_agents(self):
        """tick 后 agent 的 age 应增加。"""
        w = World(num_agents=5, num_classes=2)
        w.tick()
        for a in w.agents:
            assert a.age == 1

    def test_multiple_ticks(self):
        """多轮 tick 后 value 持续增长。"""
        w = World(num_agents=5, num_classes=2)
        w.tick()
        v1 = sum(a.value for a in w.agents)
        w.tick()
        v2 = sum(a.value for a in w.agents)
        assert v2 > v1


class TestWorldPhase1Params:
    """阶段一参数实验测试。"""

    def test_phase1_default_config(self):
        """阶段一默认配置：1000 人，5 层，指定 env。"""
        w = World.phase1()
        assert len(w.agents) == 1000
        assert w.num_classes == 5
        assert w.env_coefficients == [1.0, 1.5, 2.5, 4.0, 6.0]

    def test_phase1_agents_evenly_distributed(self):
        """阶段一每层 200 人。"""
        w = World.phase1()
        counts = [0] * 5
        for a in w.agents:
            counts[a.cls] += 1
        assert counts == [200, 200, 200, 200, 200]

    def test_phase1_ge_distribution(self):
        """阶段一 ge 分布均值约 5。"""
        import numpy as np
        w = World.phase1()
        ge_values = [a.ge for a in w.agents]
        mean_ge = np.mean(ge_values)
        assert 4.0 < mean_ge < 6.0, f"Mean ge={mean_ge}, expected ~5"

    def test_phase1_hard_distribution(self):
        """阶段一 hard 分布均值约 5。"""
        import numpy as np
        w = World.phase1()
        hard_values = [a.hard for a in w.agents]
        mean_hard = np.mean(hard_values)
        assert 4.0 < mean_hard < 6.0, f"Mean hard={mean_hard}, expected ~5"

    def test_phase1_value_starts_at_zero(self):
        """阶段一开始时所有 value 为 0。"""
        w = World.phase1()
        for a in w.agents:
            assert a.value == 0.0