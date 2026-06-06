"""Tests for Agent data structure."""
import pytest
from stratify.agent import Agent


class TestAgentCreation:
    """Agent 基本创建与属性测试。"""

    def test_create_agent_with_defaults(self):
        """创建一个 Agent，所有属性应有合理默认值。"""
        agent = Agent()
        assert agent.value == 0.0
        assert agent.alive is True
        assert agent.age == 0

    def test_create_agent_with_attributes(self):
        """通过参数创建 Agent。"""
        agent = Agent(cls=2, ge=7.0, hard=5.0, life=800.0)
        assert agent.cls == 2
        assert agent.ge == 7.0
        assert agent.hard == 5.0
        assert agent.life == 800.0
        assert agent.value == 0.0
        assert agent.alive is True
        assert agent.max_class == 2  # 初始最高阶级 = 初始阶级

    def test_agent_value_can_be_set(self):
        """Value 可以被修改。"""
        agent = Agent(cls=0, ge=5.0, hard=5.0, life=800.0)
        agent.value = 100.0
        assert agent.value == 100.0

    def test_agent_cls_must_be_non_negative(self):
        """阶级不能为负数。"""
        with pytest.raises(ValueError):
            Agent(cls=-1)

    def test_agent_ge_must_be_positive(self):
        """天赋值必须为正。"""
        with pytest.raises(ValueError):
            Agent(cls=0, ge=0.0, hard=5.0, life=800.0)

    def test_agent_hard_must_be_positive(self):
        """努力值必须为正。"""
        with pytest.raises(ValueError):
            Agent(cls=0, ge=5.0, hard=-1.0, life=800.0)

    def test_agent_life_must_be_positive(self):
        """寿命必须为正。"""
        with pytest.raises(ValueError):
            Agent(cls=0, ge=5.0, hard=5.0, life=0.0)


class TestAgentMaxClass:
    """历史最高阶级追踪测试。"""

    def test_max_class_updates_on_promotion(self):
        """阶级提升时，max_class 应自动更新。"""
        agent = Agent(cls=0, ge=5.0, hard=5.0, life=800.0)
        assert agent.max_class == 0
        agent.cls = 3
        assert agent.max_class == 3

    def test_max_class_does_not_decrease(self):
        """阶级下降时，max_class 不应降低。"""
        agent = Agent(cls=3, ge=5.0, hard=5.0, life=800.0)
        assert agent.max_class == 3
        agent.cls = 1
        assert agent.max_class == 3

    def test_max_class_tracks_highest(self):
        """多次变动后，max_class 记录历史最高。"""
        agent = Agent(cls=0, ge=5.0, hard=5.0, life=800.0)
        agent.cls = 2
        agent.cls = 4
        agent.cls = 1
        assert agent.max_class == 4