"""Tests for value accumulation formulas."""
import pytest
from stratify.accumulation import accumulate_value, accumulation_speed


class TestAccumulationSpeed:
    """积累速度公式测试。"""

    def test_basic_speed_formula(self):
        """speed = env * (ge + hard) / 2"""
        speed = accumulation_speed(env=2.0, ge=6.0, hard=4.0)
        assert speed == pytest.approx(2.0 * (6.0 + 4.0) / 2)

    def test_speed_scales_with_env(self):
        """更高阶层系数 → 更快积累。"""
        low = accumulation_speed(env=1.0, ge=5.0, hard=5.0)
        high = accumulation_speed(env=5.0, ge=5.0, hard=5.0)
        assert high == pytest.approx(low * 5.0)

    def test_speed_scales_with_ge(self):
        """更高天赋 → 更快积累。"""
        low = accumulation_speed(env=2.0, ge=2.0, hard=5.0)
        high = accumulation_speed(env=2.0, ge=8.0, hard=5.0)
        assert high > low

    def test_speed_scales_with_hard(self):
        """更高努力 → 更快积累。"""
        low = accumulation_speed(env=2.0, ge=5.0, hard=2.0)
        high = accumulation_speed(env=2.0, ge=5.0, hard=8.0)
        assert high > low

    def test_speed_with_zero_env_is_zero(self):
        """env=0 时积累速度为零。"""
        speed = accumulation_speed(env=0.0, ge=10.0, hard=10.0)
        assert speed == 0.0

    def test_speed_symmetric_in_ge_and_hard(self):
        """ge 和 hard 对称贡献。"""
        s1 = accumulation_speed(env=2.0, ge=3.0, hard=7.0)
        s2 = accumulation_speed(env=2.0, ge=7.0, hard=3.0)
        assert s1 == pytest.approx(s2)


class TestAccumulateValue:
    """单步积累测试。"""

    def test_value_increases_by_speed(self):
        """一次积累后 value 应增加 speed 的量。"""
        agent_value = 100.0
        speed = 5.0
        new_value = accumulate_value(agent_value, speed)
        assert new_value == pytest.approx(105.0)

    def test_accumulate_from_zero(self):
        """从零开始积累。"""
        new_value = accumulate_value(0.0, 3.5)
        assert new_value == pytest.approx(3.5)

    def test_accumulate_multiple_steps(self):
        """多步积累等价于 speed * steps。"""
        value = 0.0
        speed = 2.5
        for _ in range(10):
            value = accumulate_value(value, speed)
        assert value == pytest.approx(25.0)