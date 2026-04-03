"""Tests for dynamic TCP communication primitives (no torch.distributed)."""
import pytest
import io
import struct
import socket
import threading
import time
import torch

from ravnest.communication.communication_dynamic import TensorSocket, AsyncWork


class TestTensorSocket:
    def test_send_recv_tensor(self):
        """Send a tensor through a socket pair and verify it arrives intact."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen(1)

        original = torch.randn(3, 4)
        received = [None]

        def server_side():
            conn, _ = server.accept()
            received[0] = TensorSocket.recv_tensor(conn)
            conn.close()

        t = threading.Thread(target=server_side)
        t.start()

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        TensorSocket.send_tensor(client, original)
        client.close()

        t.join(timeout=5)
        server.close()

        assert received[0] is not None
        assert torch.allclose(original, received[0])

    def test_send_recv_int_tensor(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen(1)

        original = torch.tensor([42, 7, 13], dtype=torch.int64)
        received = [None]

        def server_side():
            conn, _ = server.accept()
            received[0] = TensorSocket.recv_tensor(conn)
            conn.close()

        t = threading.Thread(target=server_side)
        t.start()

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        TensorSocket.send_tensor(client, original)
        client.close()

        t.join(timeout=5)
        server.close()

        assert torch.equal(original, received[0])

    def test_send_recv_object(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen(1)

        original = {"key": "value", "list": [1, 2, 3]}
        received = [None]

        def server_side():
            conn, _ = server.accept()
            received[0] = TensorSocket.recv_object(conn)
            conn.close()

        t = threading.Thread(target=server_side)
        t.start()

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        TensorSocket.send_object(client, original)
        client.close()

        t.join(timeout=5)
        server.close()

        assert received[0] == original

    def test_large_tensor(self):
        """Test with a tensor large enough to need multiple recv calls."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen(1)

        # ~4MB tensor
        original = torch.randn(1000, 1000)
        received = [None]

        def server_side():
            conn, _ = server.accept()
            received[0] = TensorSocket.recv_tensor(conn)
            conn.close()

        t = threading.Thread(target=server_side)
        t.start()

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        TensorSocket.send_tensor(client, original)
        client.close()

        t.join(timeout=10)
        server.close()

        assert torch.allclose(original, received[0])


class TestAsyncWork:
    def test_completes(self):
        work = AsyncWork(lambda: 42)
        work.wait()
        assert work.is_completed()

    def test_not_completed_immediately(self):
        work = AsyncWork(lambda: time.sleep(0.5))
        # Might or might not be done, but shouldn't crash
        work.wait()
        assert work.is_completed()

    def test_error_propagates(self):
        def fail():
            raise ValueError("test error")

        work = AsyncWork(fail)
        with pytest.raises(ValueError, match="test error"):
            work.wait()

    def test_is_completed_false_during_work(self):
        event = threading.Event()

        def wait_for_event():
            event.wait()

        work = AsyncWork(wait_for_event)
        assert not work.is_completed()
        event.set()
        work.wait()
        assert work.is_completed()
