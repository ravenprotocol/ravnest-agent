"""Tests for coordinator cluster state logic (no server needed)."""
import pytest
import time
from ravnest.coordinator import ClusterState


class TestClusterState:
    def setup_method(self):
        self.state = ClusterState(min_nodes=2, heartbeat_timeout=5)

    def test_empty_cluster(self):
        status = self.state._status()
        assert status["num_nodes"] == 0
        assert status["ready"] is False
        assert status["cluster_version"] == 0

    def test_register_one_node(self):
        result = self.state.register("node-1", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        assert result["num_nodes"] == 1
        assert result["ready"] is False
        assert result["cluster_version"] == 1

    def test_register_two_nodes_ready(self):
        self.state.register("node-1", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        result = self.state.register("node-2", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        assert result["num_nodes"] == 2
        assert result["ready"] is True
        assert result["cluster_version"] == 2

    def test_duplicate_register_no_version_bump(self):
        self.state.register("node-1", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        v1 = self.state.cluster_version
        self.state.register("node-1", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        assert self.state.cluster_version == v1

    def test_ranks_assigned_sorted(self):
        self.state.register("node-b", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        self.state.register("node-a", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        assert self.state.nodes["node-a"]["rank"] == 0
        assert self.state.nodes["node-b"]["rank"] == 1

    def test_remove_node(self):
        self.state.register("node-1", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        self.state.register("node-2", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        result = self.state.remove("node-2")
        assert result["num_nodes"] == 1
        assert result["ready"] is False

    def test_remove_nonexistent(self):
        result = self.state.remove("ghost")
        assert result["num_nodes"] == 0

    def test_heartbeat_updates_last_seen(self):
        self.state.register("node-1", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        before = self.state.nodes["node-1"]["last_seen"]
        time.sleep(0.1)
        self.state.heartbeat("node-1")
        after = self.state.nodes["node-1"]["last_seen"]
        assert after > before

    def test_heartbeat_unknown_node_raises(self):
        with pytest.raises(KeyError):
            self.state.heartbeat("ghost")

    def test_heartbeat_timeout_removes_dead_nodes(self):
        self.state.register("node-1", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        self.state.register("node-2", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        # Fake old last_seen
        self.state.nodes["node-2"]["last_seen"] = time.time() - 100
        self.state.check_heartbeats()
        assert "node-2" not in self.state.nodes
        assert len(self.state.nodes) == 1


class TestProportions:
    def setup_method(self):
        self.state = ClusterState(min_nodes=2)

    def test_equal_cpu_proportions(self):
        self.state.register("a", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        self.state.register("b", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        assert self.state.proportions == [0.5, 0.5]

    def test_gpu_gets_more(self):
        self.state.register("a", {"type": "cuda", "name": "RTX 3090", "memory_gb": 24})
        self.state.register("b", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        assert self.state.proportions[0] > self.state.proportions[1]

    def test_proportions_sum_to_one(self):
        self.state.register("a", {"type": "cuda", "name": "RTX 3090", "memory_gb": 24})
        self.state.register("b", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        self.state.register("c", {"type": "cuda", "name": "RTX 4060", "memory_gb": 8})
        assert sum(self.state.proportions) == pytest.approx(1.0)

    def test_recompute_on_leave(self):
        self.state.register("a", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        self.state.register("b", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        self.state.register("c", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        assert len(self.state.proportions) == 3
        self.state.remove("c")
        assert len(self.state.proportions) == 2


class TestBarrier:
    def setup_method(self):
        self.state = ClusterState(min_nodes=2)
        self.state.register("a", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        self.state.register("b", {"type": "cpu", "name": "CPU", "memory_gb": 16})

    def test_barrier_not_ready_initially(self):
        v = self.state.cluster_version
        assert self.state._check_barrier(v) is False

    def test_barrier_partial_ready(self):
        v = self.state.cluster_version
        self.state.signal_ready("a", v)
        assert self.state._check_barrier(v) is False

    def test_barrier_all_ready(self):
        v = self.state.cluster_version
        self.state.signal_ready("a", v)
        self.state.signal_ready("b", v)
        assert self.state._check_barrier(v) is True

    def test_barrier_clears_on_recompute(self):
        v = self.state.cluster_version
        self.state.signal_ready("a", v)
        self.state.signal_ready("b", v)
        # New node joins, version bumps, old barrier cleared
        self.state.register("c", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        new_v = self.state.cluster_version
        assert new_v > v
        assert self.state._check_barrier(new_v) is False

    def test_barrier_with_dead_node(self):
        v = self.state.cluster_version
        self.state.signal_ready("a", v)
        # Node b dies (removed)
        self.state.remove("b")
        # Version bumped, but for the NEW version only 'a' exists
        new_v = self.state.cluster_version
        self.state.signal_ready("a", new_v)
        assert self.state._check_barrier(new_v) is True


class TestEvents:
    def setup_method(self):
        self.state = ClusterState(min_nodes=2)

    def test_join_event(self):
        self.state.register("a", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        assert len(self.state.events) == 1
        assert self.state.events[0]["type"] == "join"
        assert self.state.events[0]["node"] == "a"

    def test_leave_event(self):
        self.state.register("a", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        self.state.remove("a")
        assert self.state.events[-1]["type"] == "leave"

    def test_timeout_event(self):
        self.state.register("a", {"type": "cpu", "name": "CPU", "memory_gb": 16})
        self.state.nodes["a"]["last_seen"] = 0
        self.state.check_heartbeats()
        assert self.state.events[-1]["type"] == "timeout"
