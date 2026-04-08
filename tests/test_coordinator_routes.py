"""Tests for the coordinator FastAPI routes (not just ClusterState class)."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from ravnest.coordinator import create_coordinator_app


@pytest.fixture
def app():
    return create_coordinator_app(min_nodes=2, heartbeat_timeout=60)


@pytest.fixture
def client(app):
    return TestClient(app)


def make_node(node_id, hw_type="cpu", memory_gb=16.0, name="CPU"):
    return {
        "node_id": node_id,
        "hardware": {"type": hw_type, "memory_gb": memory_gb, "name": name},
    }


class TestStatusRoute:
    def test_empty_cluster_status(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_nodes"] == 0
        assert data["ready"] is False
        assert data["cluster_version"] == 0

    def test_status_after_register(self, client):
        client.post("/register", json=make_node("node-a"))
        resp = client.get("/status")
        data = resp.json()
        assert data["num_nodes"] == 1
        assert data["cluster_version"] == 1

    def test_status_includes_min_nodes(self, client):
        resp = client.get("/status")
        assert resp.json()["min_nodes"] == 2

    def test_status_ready_when_min_reached(self, client):
        client.post("/register", json=make_node("node-a"))
        client.post("/register", json=make_node("node-b"))
        resp = client.get("/status")
        assert resp.json()["ready"] is True


class TestRegisterRoute:
    def test_register_returns_status(self, client):
        resp = client.post("/register", json=make_node("node-a"))
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_nodes"] == 1

    def test_register_two_nodes_makes_ready(self, client):
        client.post("/register", json=make_node("node-a"))
        resp = client.post("/register", json=make_node("node-b"))
        assert resp.json()["ready"] is True

    def test_register_invalid_payload(self, client):
        # Missing required fields
        resp = client.post("/register", json={"node_id": "x"})  # missing hardware
        assert resp.status_code == 422  # FastAPI validation error

    def test_register_idempotent(self, client):
        client.post("/register", json=make_node("node-a"))
        client.post("/register", json=make_node("node-a"))  # same node again
        resp = client.get("/status")
        assert resp.json()["num_nodes"] == 1  # not duplicated


class TestHeartbeatRoute:
    def test_heartbeat_known_node(self, client):
        client.post("/register", json=make_node("node-a"))
        resp = client.post("/heartbeat/node-a")
        assert resp.status_code == 200

    def test_heartbeat_unknown_node_404(self, client):
        resp = client.post("/heartbeat/ghost")
        assert resp.status_code == 404
        assert "not registered" in resp.json()["detail"].lower()


class TestLeaveRoute:
    def test_leave_removes_node(self, client):
        client.post("/register", json=make_node("node-a"))
        client.post("/register", json=make_node("node-b"))
        resp = client.post("/leave/node-a")
        assert resp.status_code == 200
        assert resp.json()["num_nodes"] == 1

    def test_leave_unknown_node_no_error(self, client):
        # Leave on a non-existent node is a no-op (idempotent)
        resp = client.post("/leave/ghost")
        assert resp.status_code == 200


class TestConfigRoute:
    def test_config_404_for_unknown_node(self, client):
        client.post("/register", json=make_node("node-a"))
        client.post("/register", json=make_node("node-b"))
        resp = client.get("/config/ghost")
        assert resp.status_code == 404

    def test_config_503_when_not_ready(self, client):
        client.post("/register", json=make_node("node-a"))  # only 1 node
        resp = client.get("/config/node-a")
        assert resp.status_code == 503

    def test_config_returns_rank_and_world(self, client):
        client.post("/register", json=make_node("node-a"))
        client.post("/register", json=make_node("node-b"))
        resp = client.get("/config/node-a")
        assert resp.status_code == 200
        data = resp.json()
        assert "rank" in data
        assert data["world_size"] == 2

    def test_config_includes_proportions(self, client):
        client.post("/register", json=make_node("node-a"))
        client.post("/register", json=make_node("node-b"))
        resp = client.get("/config/node-a")
        data = resp.json()
        assert "proportions" in data
        assert len(data["proportions"]) == 2

    def test_config_includes_peers(self, client):
        client.post("/register", json=make_node("node-a"))
        client.post("/register", json=make_node("node-b"))
        resp = client.get("/config/node-a")
        data = resp.json()
        assert "peers" in data
        # peers maps rank (as string in JSON) to node_id
        assert "0" in data["peers"] or 0 in data["peers"]

    def test_config_master_addr_is_first_node(self, client):
        client.post("/register", json=make_node("node-b"))
        client.post("/register", json=make_node("node-a"))  # sorts before node-b
        resp = client.get("/config/node-a")
        data = resp.json()
        # Sorted alphabetically, node-a is first
        assert data["master_addr"] == "node-a"


class TestSignalReadyRoute:
    def test_signal_ready_returns_count(self, client):
        client.post("/register", json=make_node("node-a"))
        client.post("/register", json=make_node("node-b"))
        # Get current version
        v = client.get("/status").json()["cluster_version"]
        resp = client.post(f"/ready/node-a/{v}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready_count"] == 1
        assert data["total_needed"] == 2
        assert data["all_ready"] is False

    def test_all_ready_when_all_signal(self, client):
        client.post("/register", json=make_node("node-a"))
        client.post("/register", json=make_node("node-b"))
        v = client.get("/status").json()["cluster_version"]
        client.post(f"/ready/node-a/{v}")
        resp = client.post(f"/ready/node-b/{v}")
        assert resp.json()["all_ready"] is True

    def test_signal_unknown_node_404(self, client):
        resp = client.post("/ready/ghost/1")
        assert resp.status_code == 404


class TestBarrierRoute:
    def test_barrier_initially_not_ready(self, client):
        client.post("/register", json=make_node("node-a"))
        client.post("/register", json=make_node("node-b"))
        v = client.get("/status").json()["cluster_version"]
        resp = client.get(f"/barrier/{v}")
        assert resp.status_code == 200
        assert resp.json()["all_ready"] is False

    def test_barrier_ready_after_signals(self, client):
        client.post("/register", json=make_node("node-a"))
        client.post("/register", json=make_node("node-b"))
        v = client.get("/status").json()["cluster_version"]
        client.post(f"/ready/node-a/{v}")
        client.post(f"/ready/node-b/{v}")
        resp = client.get(f"/barrier/{v}")
        assert resp.json()["all_ready"] is True

    def test_barrier_clears_on_new_version(self, client):
        client.post("/register", json=make_node("node-a"))
        client.post("/register", json=make_node("node-b"))
        v1 = client.get("/status").json()["cluster_version"]
        client.post(f"/ready/node-a/{v1}")
        client.post(f"/ready/node-b/{v1}")

        # Add a third node, version bumps
        client.post("/register", json=make_node("node-c"))
        v2 = client.get("/status").json()["cluster_version"]
        assert v2 > v1
        resp = client.get(f"/barrier/{v2}")
        assert resp.json()["all_ready"] is False


class TestDashboardRoute:
    def test_dashboard_returns_html(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Ravnest Cluster Dashboard" in resp.text

    def test_dashboard_shows_zero_nodes_initially(self, client):
        resp = client.get("/dashboard")
        assert "No nodes connected" in resp.text or "Nodes" in resp.text

    def test_dashboard_shows_registered_nodes(self, client):
        client.post("/register", json=make_node("node-a", hw_type="cuda", name="RTX 3090", memory_gb=24))
        client.post("/register", json=make_node("node-b"))
        resp = client.get("/dashboard")
        assert "node-a" in resp.text
        assert "node-b" in resp.text
        assert "RTX 3090" in resp.text

    def test_dashboard_includes_event_log(self, client):
        client.post("/register", json=make_node("node-a"))
        resp = client.get("/dashboard")
        assert "Recent Events" in resp.text
