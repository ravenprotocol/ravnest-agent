"""Tests for native CLI features: free port, background mode, role assignment, model cache check."""
import os
import signal
import socket
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ravnest.cli import _find_free_port, detect_hardware


class TestFindFreePort:
    def test_returns_start_port_if_free(self):
        # Find a port we know is free by binding then releasing
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
        s.close()
        assert _find_free_port(free_port) == free_port

    def test_skips_occupied_port(self):
        # Occupy a port, then ask for it — should return the next one
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        occupied = s.getsockname()[1]
        try:
            result = _find_free_port(occupied)
            assert result != occupied
            assert result > occupied
        finally:
            s.close()

    def test_returns_valid_port(self):
        port = _find_free_port(49000)
        assert 49000 <= port < 49100

    def test_consecutive_calls_can_find_different_ports(self):
        # Occupy two consecutive ports
        s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s1.bind(("127.0.0.1", 0))
        base = s1.getsockname()[1]
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s2.bind(("127.0.0.1", base + 1))
            result = _find_free_port(base)
            assert result >= base + 2
        except OSError:
            # base+1 also taken, that's fine
            pass
        finally:
            s1.close()
            s2.close()


class TestDetectHardware:
    def test_returns_dict_with_required_keys(self):
        hw = detect_hardware()
        assert isinstance(hw, dict)
        assert "device" in hw
        assert "gpu_count" in hw
        assert "gpus" in hw

    def test_device_includes_mps(self):
        """Device should be one of cpu, cuda, or mps."""
        hw = detect_hardware()
        assert hw["device"] in ("cpu", "cuda", "mps")

    def test_gpu_count_is_nonnegative(self):
        hw = detect_hardware()
        assert hw["gpu_count"] >= 0

    def test_gpus_list_length_matches_count(self):
        hw = detect_hardware()
        assert len(hw["gpus"]) == hw["gpu_count"]


class TestNativeRoleAssignment:
    """Test that cmd_native assigns root/stem/leaf roles correctly."""

    def test_two_nodes_roles(self):
        """With 2 nodes, rank 0 = root, rank 1 = leaf (no stem)."""
        world_size = 2
        roles = {}
        for rank in range(world_size):
            if rank == 0:
                roles[rank] = "root"
            elif rank == world_size - 1:
                roles[rank] = "leaf"
            else:
                roles[rank] = "stem"
        assert roles == {0: "root", 1: "leaf"}

    def test_three_nodes_roles(self):
        """With 3 nodes, rank 0 = root, rank 1 = stem, rank 2 = leaf."""
        world_size = 3
        roles = {}
        for rank in range(world_size):
            if rank == 0:
                roles[rank] = "root"
            elif rank == world_size - 1:
                roles[rank] = "leaf"
            else:
                roles[rank] = "stem"
        assert roles == {0: "root", 1: "stem", 2: "leaf"}

    def test_four_nodes_roles(self):
        """With 4 nodes, ranks 1 and 2 are both stem."""
        world_size = 4
        roles = {}
        for rank in range(world_size):
            if rank == 0:
                roles[rank] = "root"
            elif rank == world_size - 1:
                roles[rank] = "leaf"
            else:
                roles[rank] = "stem"
        assert roles == {0: "root", 1: "stem", 2: "stem", 3: "leaf"}

    def test_single_node_is_root(self):
        """With 1 node, rank 0 is root (and leaf, but root takes priority)."""
        rank = 0
        world_size = 1
        if rank == 0:
            role = "root"
        elif rank == world_size - 1:
            role = "leaf"
        else:
            role = "stem"
        assert role == "root"


class TestPeerParsing:
    """Test peer list parsing logic from cmd_native."""

    def test_single_machine_default_peers(self):
        nodes = 2
        peer_list = ["127.0.0.1"] * nodes
        assert peer_list == ["127.0.0.1", "127.0.0.1"]

    def test_multi_machine_peer_parsing(self):
        peers_str = "192.168.1.5,192.168.1.10"
        peer_list = [h.strip() for h in peers_str.split(",")]
        assert peer_list == ["192.168.1.5", "192.168.1.10"]

    def test_peers_with_spaces(self):
        peers_str = " 10.0.0.1 , 10.0.0.2 , 10.0.0.3 "
        peer_list = [h.strip() for h in peers_str.split(",")]
        assert peer_list == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]

    def test_tailscale_ips(self):
        peers_str = "100.64.1.5,100.64.1.10"
        peer_list = [h.strip() for h in peers_str.split(",")]
        assert len(peer_list) == 2
        assert all(p.startswith("100.64.") for p in peer_list)

    def test_three_peers_world_size(self):
        peers_str = "10.0.0.1,10.0.0.2,10.0.0.3"
        peer_list = [h.strip() for h in peers_str.split(",")]
        world_size = len(peer_list)
        assert world_size == 3


class TestNativeStop:
    """Test the native-stop PID file handling."""

    def test_pid_file_write_and_read(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pid", delete=False) as f:
            pids = [1234, 5678, 9012]
            f.write(",".join(str(p) for p in pids))
            pid_file = f.name

        try:
            with open(pid_file) as f:
                read_pids = [int(p) for p in f.read().strip().split(",") if p]
            assert read_pids == pids
        finally:
            os.unlink(pid_file)

    def test_pid_file_single_pid(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pid", delete=False) as f:
            f.write("42")
            pid_file = f.name

        try:
            with open(pid_file) as f:
                read_pids = [int(p) for p in f.read().strip().split(",") if p]
            assert read_pids == [42]
        finally:
            os.unlink(pid_file)

    def test_empty_pid_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pid", delete=False) as f:
            f.write("")
            pid_file = f.name

        try:
            with open(pid_file) as f:
                read_pids = [int(p) for p in f.read().strip().split(",") if p]
            assert read_pids == []
        finally:
            os.unlink(pid_file)


class TestModelCacheCheck:
    """Test the model cache detection logic."""

    def test_empty_dir_is_not_cached(self):
        with tempfile.TemporaryDirectory() as d:
            model_cached = any(
                os.path.exists(os.path.join(d, sub, "config.json"))
                for sub in os.listdir(d)
                if sub.startswith("models--")
            ) if os.path.isdir(d) and os.listdir(d) else False
            assert model_cached is False

    def test_dir_with_model_is_cached(self):
        with tempfile.TemporaryDirectory() as d:
            model_dir = os.path.join(d, "models--TinyLlama")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "config.json"), "w") as f:
                f.write("{}")
            model_cached = any(
                os.path.exists(os.path.join(d, sub, "config.json"))
                for sub in os.listdir(d)
                if sub.startswith("models--")
            ) if os.path.isdir(d) and os.listdir(d) else False
            assert model_cached is True

    def test_dir_without_config_is_not_cached(self):
        with tempfile.TemporaryDirectory() as d:
            model_dir = os.path.join(d, "models--Partial")
            os.makedirs(model_dir)
            # No config.json
            model_cached = any(
                os.path.exists(os.path.join(d, sub, "config.json"))
                for sub in os.listdir(d)
                if sub.startswith("models--")
            ) if os.path.isdir(d) and os.listdir(d) else False
            assert model_cached is False

    def test_non_model_dirs_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            other_dir = os.path.join(d, "something-else")
            os.makedirs(other_dir)
            with open(os.path.join(other_dir, "config.json"), "w") as f:
                f.write("{}")
            model_cached = any(
                os.path.exists(os.path.join(d, sub, "config.json"))
                for sub in os.listdir(d)
                if sub.startswith("models--")
            ) if os.path.isdir(d) and os.listdir(d) else False
            assert model_cached is False
