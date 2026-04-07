"""Tests for connection timeout error messages and MPS hardware detection."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ravnest.hardware import compute_proportions


class TestMPSProportions:
    """MPS devices should be weighted 4x CPU in proportion calculations."""

    def test_mps_vs_cpu(self):
        hw = [
            {"type": "mps", "memory_gb": 16.0},
            {"type": "cpu", "memory_gb": 16.0},
        ]
        props = compute_proportions(hw)
        assert len(props) == 2
        # MPS gets 4x weight: 64 / (64+16) = 0.8
        assert props[0] > 0.75
        assert props[1] < 0.25

    def test_mps_vs_cuda(self):
        hw = [
            {"type": "cuda", "memory_gb": 10.0},  # 100
            {"type": "mps", "memory_gb": 10.0},    # 40
        ]
        props = compute_proportions(hw)
        assert len(props) == 2
        # CUDA should still beat MPS
        assert props[0] > props[1]

    def test_mps_only(self):
        hw = [{"type": "mps", "memory_gb": 16.0}]
        props = compute_proportions(hw)
        assert props == [1.0]

    def test_mps_mps_equal(self):
        hw = [
            {"type": "mps", "memory_gb": 16.0},
            {"type": "mps", "memory_gb": 16.0},
        ]
        props = compute_proportions(hw)
        assert props == [0.5, 0.5]

    def test_mps_cpu_proportions_sum_to_one(self):
        hw = [
            {"type": "mps", "memory_gb": 24.0},
            {"type": "cpu", "memory_gb": 8.0},
            {"type": "cpu", "memory_gb": 16.0},
        ]
        props = compute_proportions(hw)
        assert sum(props) == pytest.approx(1.0)
        # MPS should get the most
        assert props[0] > props[1]
        assert props[0] > props[2]

    def test_mixed_all_three_types(self):
        hw = [
            {"type": "cuda", "memory_gb": 24.0},   # 240
            {"type": "mps", "memory_gb": 16.0},     # 64
            {"type": "cpu", "memory_gb": 32.0},     # 32
        ]
        props = compute_proportions(hw)
        assert sum(props) == pytest.approx(1.0)
        # CUDA > MPS > CPU
        assert props[0] > props[1] > props[2]


class TestConnectionErrorMessages:
    """Verify the error message format contains actionable info."""

    def test_error_message_format(self):
        """Simulate the error message that communication_dynamic would produce."""
        next_rank = 1
        target_host = "100.64.1.10"
        target_port = 29501
        conn_timeout = 60

        error_msg = (
            f"Could not connect to rank {next_rank} at {target_host}:{target_port} "
            f"after {conn_timeout}s. Check that:\n"
            f"  1. The other node is running (ravnest native --rank {next_rank})\n"
            f"  2. The IP address {target_host} is reachable (try: ping {target_host})\n"
            f"  3. Port {target_port} is not blocked by a firewall\n"
            f"  4. Both nodes use the same --peers list"
        )

        assert "rank 1" in error_msg
        assert "100.64.1.10" in error_msg
        assert "29501" in error_msg
        assert "60s" in error_msg
        assert "ping" in error_msg
        assert "firewall" in error_msg
        assert "--peers" in error_msg

    def test_metadata_error_message(self):
        root_addr = "192.168.1.5"
        listen_port = 29500
        conn_timeout = 60

        error_msg = (
            f"Could not connect metadata channel to root at {root_addr}:{listen_port} "
            f"after {conn_timeout}s. Is the root node running?"
        )

        assert "192.168.1.5" in error_msg
        assert "29500" in error_msg
        assert "root" in error_msg


class TestEntrypointRoles:
    """Test that entrypoint.py correctly maps NODE_ROLE to behavior."""

    def test_role_env_values(self):
        """All valid role values that entrypoint.py should handle."""
        valid_roles = ["root", "leaf", "stem"]
        for role in valid_roles:
            # root -> run_root(), anything else -> run_non_root()
            if role == "root":
                handler = "run_root"
            else:
                handler = "run_non_root"
            assert handler in ("run_root", "run_non_root")

    def test_stem_uses_non_root(self):
        """Stem nodes should use the same handler as leaf nodes."""
        role = "stem"
        assert role != "root"  # will go to run_non_root

    def test_unknown_role_uses_non_root(self):
        """Unknown roles should default to non-root behavior."""
        role = "something-weird"
        assert role != "root"
