"""Tests for hardware profiling and proportion computation."""
import pytest
from ravnest.hardware import compute_proportions, get_hardware_info


class TestComputeProportions:
    def test_equal_cpu_nodes(self):
        hw = [
            {"type": "cpu", "memory_gb": 16.0},
            {"type": "cpu", "memory_gb": 16.0},
        ]
        props = compute_proportions(hw)
        assert len(props) == 2
        assert props == [0.5, 0.5]

    def test_gpu_weighted_10x_over_cpu(self):
        hw = [
            {"type": "cuda", "memory_gb": 10.0},  # weight = 100
            {"type": "cpu", "memory_gb": 10.0},    # weight = 10
        ]
        props = compute_proportions(hw)
        assert len(props) == 2
        # GPU should get ~91%, CPU ~9%
        assert props[0] > 0.85
        assert props[1] < 0.15

    def test_mixed_gpu_cluster(self):
        hw = [
            {"type": "cuda", "memory_gb": 24.0},  # RTX 3090
            {"type": "cpu", "memory_gb": 16.0},    # laptop
            {"type": "cuda", "memory_gb": 8.0},    # RTX 4060
        ]
        props = compute_proportions(hw)
        assert len(props) == 3
        assert sum(props) == pytest.approx(1.0)
        # 3090 should get the most
        assert props[0] > props[1]
        assert props[0] > props[2]
        # CPU should get the least
        assert props[1] < props[2]

    def test_single_node(self):
        hw = [{"type": "cuda", "memory_gb": 8.0}]
        props = compute_proportions(hw)
        assert props == [1.0]

    def test_zero_memory_fallback(self):
        hw = [
            {"type": "cpu", "memory_gb": 0},
            {"type": "cpu", "memory_gb": 0},
        ]
        props = compute_proportions(hw)
        assert len(props) == 2
        assert props == [0.5, 0.5]

    def test_many_nodes(self):
        hw = [{"type": "cpu", "memory_gb": 8.0} for _ in range(10)]
        props = compute_proportions(hw)
        assert len(props) == 10
        assert sum(props) == pytest.approx(1.0)
        assert all(p == 0.1 for p in props)

    def test_proportions_sum_to_one(self):
        hw = [
            {"type": "cuda", "memory_gb": 24.0},
            {"type": "cuda", "memory_gb": 8.0},
            {"type": "cpu", "memory_gb": 32.0},
            {"type": "cuda", "memory_gb": 12.0},
        ]
        props = compute_proportions(hw)
        assert sum(props) == pytest.approx(1.0)


class TestGetHardwareInfo:
    def test_returns_dict(self):
        info = get_hardware_info()
        assert isinstance(info, dict)
        assert "type" in info
        assert "memory_gb" in info
        assert "name" in info

    def test_type_is_valid(self):
        info = get_hardware_info()
        assert info["type"] in ("cpu", "cuda")

    def test_memory_is_positive(self):
        info = get_hardware_info()
        assert info["memory_gb"] > 0
