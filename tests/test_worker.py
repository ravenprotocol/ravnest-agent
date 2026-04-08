"""Tests for the Worker class — registration, heartbeat, config, leave."""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# Patch torch.cuda.is_available BEFORE importing Worker
class MockResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")


@pytest.fixture
def worker():
    from ravnest.worker import Worker
    with patch("ravnest.worker.get_node_id", return_value="test-node-1"), \
         patch("ravnest.worker.get_hardware_info", return_value={"type": "cpu", "memory_gb": 16.0, "name": "TestCPU"}):
        w = Worker(coordinator_url="http://fake-coordinator:8080", device="cpu")
        return w


class TestWorkerInit:
    def test_node_id_set(self, worker):
        assert worker.node_id == "test-node-1"

    def test_hardware_collected(self, worker):
        assert worker.hardware["type"] == "cpu"
        assert worker.hardware["memory_gb"] == 16.0

    def test_coordinator_url_normalized(self):
        from ravnest.worker import Worker
        with patch("ravnest.worker.get_node_id", return_value="x"), \
             patch("ravnest.worker.get_hardware_info", return_value={"type": "cpu", "memory_gb": 1, "name": "x"}):
            w = Worker(coordinator_url="http://x:8080/", device="cpu")
            assert w.coordinator_url == "http://x:8080"  # trailing slash stripped

    def test_initial_state(self, worker):
        assert worker.current_version == -1
        assert worker.running is True
        assert worker.full_model is None
        assert worker.engine is None
        assert worker.node is None

    def test_device_str_set(self, worker):
        assert worker.device_str == "cpu"


class TestRegister:
    def test_register_success(self, worker):
        with patch("requests.post") as mock_post:
            mock_post.return_value = MockResponse({
                "num_nodes": 1, "min_nodes": 2, "ready": False, "cluster_version": 1,
            })
            result = worker.register()
            assert result["num_nodes"] == 1
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/register" in call_args[0][0]
            assert call_args[1]["json"]["node_id"] == "test-node-1"

    def test_register_sends_hardware(self, worker):
        with patch("requests.post") as mock_post:
            mock_post.return_value = MockResponse({
                "num_nodes": 1, "min_nodes": 2, "ready": False, "cluster_version": 1,
            })
            worker.register()
            sent = mock_post.call_args[1]["json"]
            assert "hardware" in sent
            assert sent["hardware"]["type"] == "cpu"


class TestHeartbeat:
    def test_heartbeat_success(self, worker):
        with patch("requests.post") as mock_post:
            mock_post.return_value = MockResponse({
                "num_nodes": 2, "ready": True, "cluster_version": 1,
            })
            result = worker.heartbeat()
            assert result is not None
            assert result["ready"] is True

    def test_heartbeat_404_triggers_reregister(self, worker):
        """When coordinator returns 404, worker should re-register."""
        call_log = []

        def mock_post(url, **kwargs):
            call_log.append(url)
            if "/heartbeat" in url:
                return MockResponse({"detail": "not registered"}, status_code=404)
            elif "/register" in url:
                return MockResponse({"num_nodes": 1, "ready": False, "cluster_version": 2, "min_nodes": 2})
            return MockResponse({}, 200)

        with patch("requests.post", side_effect=mock_post):
            result = worker.heartbeat()
            # Should have made 2 calls: heartbeat (404), then register
            assert any("/heartbeat" in u for u in call_log)
            assert any("/register" in u for u in call_log)

    def test_heartbeat_connection_error_returns_none(self, worker):
        import requests
        with patch("requests.post", side_effect=requests.ConnectionError("dead")):
            result = worker.heartbeat()
            assert result is None

    def test_heartbeat_url_format(self, worker):
        with patch("requests.post") as mock_post:
            mock_post.return_value = MockResponse({"num_nodes": 1, "ready": False, "cluster_version": 1, "min_nodes": 2})
            worker.heartbeat()
            call_args = mock_post.call_args
            assert "/heartbeat/test-node-1" in call_args[0][0]


class TestGetConfig:
    def test_config_success(self, worker):
        with patch("requests.get") as mock_get:
            mock_get.return_value = MockResponse({
                "rank": 0,
                "world_size": 2,
                "master_addr": "test-node-1",
                "proportions": [0.5, 0.5],
                "cluster_version": 1,
                "model": "TinyLlama",
                "device": "cpu",
                "peers": {"0": "test-node-1", "1": "test-node-2"},
            })
            config = worker.get_config()
            assert config["rank"] == 0
            assert config["world_size"] == 2

    def test_config_503_returns_none(self, worker):
        with patch("requests.get") as mock_get:
            mock_get.return_value = MockResponse(
                {"detail": "not ready"}, status_code=503
            )
            result = worker.get_config()
            assert result is None

    def test_config_url_format(self, worker):
        with patch("requests.get") as mock_get:
            mock_get.return_value = MockResponse({"rank": 0, "world_size": 1, "master_addr": "x", "proportions": [1.0], "cluster_version": 1, "model": None, "device": None, "peers": {}})
            worker.get_config()
            call_args = mock_get.call_args
            assert "/config/test-node-1" in call_args[0][0]


class TestLeave:
    def test_leave_calls_endpoint(self, worker):
        with patch("requests.post") as mock_post:
            mock_post.return_value = MockResponse({}, 200)
            worker.leave()
            mock_post.assert_called_once()
            assert "/leave/test-node-1" in mock_post.call_args[0][0]

    def test_leave_swallows_errors(self, worker):
        """Leave should not raise on connection failure (best-effort cleanup)."""
        import requests
        with patch("requests.post", side_effect=requests.ConnectionError("dead")):
            # Should not raise
            worker.leave()


class TestConnectionErrorDetection:
    def test_detects_broken_pipe(self, worker):
        assert worker.is_connection_error(BrokenPipeError("broken pipe")) is True

    def test_detects_connection_reset(self, worker):
        assert worker.is_connection_error(Exception("Connection reset by peer")) is True

    def test_detects_timeout(self, worker):
        assert worker.is_connection_error(Exception("Connection timed out")) is True

    def test_detects_eof(self, worker):
        assert worker.is_connection_error(Exception("EOF while reading")) is True

    def test_ignores_other_errors(self, worker):
        assert worker.is_connection_error(ValueError("bad input")) is False
        assert worker.is_connection_error(KeyError("missing")) is False


class TestGetNodeId:
    def test_returns_string(self):
        from ravnest.worker import get_node_id
        result = get_node_id()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_lan_ip_fallback(self):
        """When tailscale isn't available, falls back to LAN IP."""
        from ravnest.worker import get_node_id
        # Just verify it doesn't crash and returns something
        result = get_node_id()
        assert result is not None
