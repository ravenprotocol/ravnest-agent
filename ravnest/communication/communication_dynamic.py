"""
Dynamic TCP-based communication backend for distributed inference.

Drop-in replacement for Communication_Torch that uses raw TCP sockets
instead of torch.distributed. Connections can be added/removed at runtime,
enabling hot-swap of nodes without restarting the cluster.

Protocol:
  Header: 4 bytes (message length as uint32, big-endian)
  Body: serialized tensor bytes (torch.save format)
"""

import io
import json
import os
import pickle
import socket
import struct
import threading
import time
from typing import Dict, List, Optional, Tuple

import torch

try:
    from ..strings import NodeTypes, NodeModes
except ImportError:
    from ravnest.strings import NodeTypes, NodeModes


class TensorSocket:
    """Send and receive PyTorch tensors over a TCP socket."""

    @staticmethod
    def send_tensor(sock, tensor):
        """Send a tensor over a socket with length-prefixed framing."""
        buf = io.BytesIO()
        torch.save(tensor, buf)
        data = buf.getvalue()
        header = struct.pack(">I", len(data))
        sock.sendall(header + data)

    @staticmethod
    def recv_tensor(sock, device="cpu", timeout=300):
        """Receive a tensor from a socket. Pass timeout=None to block indefinitely."""
        sock.settimeout(timeout)  # None = blocking forever
        header = TensorSocket._recv_exactly(sock, 4)
        if header is None:
            raise ConnectionError("Connection closed while reading header")
        length = struct.unpack(">I", header)[0]
        data = TensorSocket._recv_exactly(sock, length)
        if data is None:
            raise ConnectionError("Connection closed while reading tensor")
        buf = io.BytesIO(data)
        tensor = torch.load(buf, map_location=device, weights_only=True)
        return tensor

    @staticmethod
    def send_object(sock, obj):
        """Send a Python object (for metadata)."""
        data = pickle.dumps(obj)
        header = struct.pack(">I", len(data))
        sock.sendall(header + data)

    @staticmethod
    def recv_object(sock, timeout=300):
        """Receive a Python object."""
        sock.settimeout(timeout)
        header = TensorSocket._recv_exactly(sock, 4)
        if header is None:
            raise ConnectionError("Connection closed while reading header")
        length = struct.unpack(">I", header)[0]
        data = TensorSocket._recv_exactly(sock, length)
        if data is None:
            raise ConnectionError("Connection closed while reading object")
        return pickle.loads(data)

    @staticmethod
    def _recv_exactly(sock, n):
        """Receive exactly n bytes."""
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)


class AsyncWork:
    """Mimics torch.distributed Work object for compatibility."""

    def __init__(self, fn, args=()):
        self._done = threading.Event()
        self._result = None
        self._error = None
        self._thread = threading.Thread(target=self._run, args=(fn, args), daemon=True)
        self._thread.start()

    def _run(self, fn, args):
        try:
            self._result = fn(*args)
        except Exception as e:
            self._error = e
        finally:
            self._done.set()

    def is_completed(self):
        return self._done.is_set()

    def wait(self):
        self._done.wait()
        if self._error:
            raise self._error


class Communication_Dynamic:
    """Dynamic TCP-based communication for distributed inference.

    Same interface as Communication_Torch so Node and InferenceEngine
    work without changes. Connections are managed via connect/accept
    rather than init_process_group.
    """

    def __init__(self, rank=0, world_size=2, node_type=None,
                 mode=NodeModes.INFERENCE,
                 forward_input_shapes=None, feedback_shape=None,
                 backward_input_shapes=None,
                 dtype=None, device=None, input_tensors=None,
                 peer_addresses=None, peer_ips=None, listen_port=29500):
        self.rank = rank
        self.world_size = world_size
        self.mode = mode
        self.dtype = dtype
        self.device = device
        self.input_tensors = input_tensors
        self.forward_input_shapes = forward_input_shapes
        self.backward_input_shapes = backward_input_shapes
        self.feedback_shape = feedback_shape
        self.listen_port = listen_port
        # Build peer_ips from explicit param, env var, or empty
        if peer_ips:
            self.peer_ips = peer_ips
        else:
            # Parse RAVNEST_PEERS env: "host0,host1,host2" -> {0: host0, 1: host1, ...}
            peers_env = os.environ.get("RAVNEST_PEERS", "")
            if peers_env:
                self.peer_ips = {i: h for i, h in enumerate(peers_env.split(","))}
            else:
                self.peer_ips = {}

        if node_type is not None:
            self.node_type = node_type
        elif rank == 0:
            self.node_type = NodeTypes.ROOT
        elif rank == world_size - 1:
            self.node_type = NodeTypes.LEAF
        else:
            self.node_type = NodeTypes.STEM

        # Connection state
        self.peers: Dict[int, socket.socket] = {}  # rank -> socket
        self._server_sock = None
        self._lock = threading.Lock()
        self.is_receiving_fwd = False

        # Async work placeholders
        self.forward_send_work = None
        self.backward_send_work = None
        self.feedback_send_work = None
        self.forward_recv_works = None
        self.backward_recv_works = None
        self.feedback_recv_work = None

        # Forward/backward buffers
        self.forward_ips = None
        self.backward_ips = None
        self.feedback_ip = None

        if peer_addresses:
            self._connect_to_peers(peer_addresses)
        else:
            self._setup_connections()

    def _setup_connections(self):
        """Establish connections between pipeline-adjacent nodes."""
        # Listen for incoming connections
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(("0.0.0.0", self.listen_port + self.rank))
        self._server_sock.listen(self.world_size)
        self._server_sock.settimeout(300)

        print(f"[dynamic-comm] Rank {self.rank} listening on port {self.listen_port + self.rank}")

        # Connection protocol: lower rank connects to higher rank
        # Higher rank accepts from lower rank
        threads = []
        conn_timeout = int(os.environ.get("RAVNEST_CONN_TIMEOUT", "60"))
        max_attempts = conn_timeout // 2  # retry every 2 seconds

        # Accept from lower rank (if not ROOT)
        if self.rank > 0:
            def accept_prev():
                conn, addr = self._server_sock.accept()
                # Read the sender's rank
                rank_data = conn.recv(4)
                sender_rank = struct.unpack(">I", rank_data)[0]
                self.peers[sender_rank] = conn
                print(f"[dynamic-comm] Rank {self.rank} accepted connection from rank {sender_rank} ({addr})")
            t = threading.Thread(target=accept_prev, daemon=True)
            t.start()
            threads.append(t)

        # Connect to next rank (if not LEAF)
        if self.rank < self.world_size - 1:
            def connect_next():
                next_rank = self.rank + 1
                target_host = self.peer_ips.get(next_rank, os.environ.get("MASTER_ADDR", "localhost"))
                target_port = self.listen_port + next_rank
                for attempt in range(max_attempts):
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(5)
                        sock.connect((target_host, target_port))
                        sock.settimeout(None)
                        sock.sendall(struct.pack(">I", self.rank))
                        self.peers[next_rank] = sock
                        print(f"[dynamic-comm] Rank {self.rank} connected to rank {next_rank} ({target_host}:{target_port})")
                        return
                    except (ConnectionRefusedError, OSError) as e:
                        if attempt % 5 == 0 and attempt > 0:
                            print(f"[dynamic-comm] Rank {self.rank} still waiting for rank {next_rank} at {target_host}:{target_port} ({attempt * 2}s elapsed)...")
                        time.sleep(2)
                raise RuntimeError(
                    f"Could not connect to rank {next_rank} at {target_host}:{target_port} "
                    f"after {conn_timeout}s. Check that:\n"
                    f"  1. The other node is running (ravnest native --rank {next_rank})\n"
                    f"  2. The IP address {target_host} is reachable (try: ping {target_host})\n"
                    f"  3. Port {target_port} is not blocked by a firewall\n"
                    f"  4. Both nodes use the same --peers list"
                )
            t = threading.Thread(target=connect_next, daemon=True)
            t.start()
            threads.append(t)

        # Also need metadata/feedback channels (all-to-root)
        if self.rank == 0:
            # Root accepts metadata connections from all other ranks
            def accept_metadata():
                for _ in range(self.world_size - 1):
                    conn, addr = self._server_sock.accept()
                    rank_data = conn.recv(4)
                    sender_rank = struct.unpack(">I", rank_data)[0]
                    key = f"meta_{sender_rank}"
                    self.peers[key] = conn
                    print(f"[dynamic-comm] Rank 0 metadata channel from rank {sender_rank}")
            t = threading.Thread(target=accept_metadata, daemon=True)
            t.start()
            threads.append(t)
        else:
            # Non-root connects to root for metadata
            def connect_metadata():
                root_addr = self.peer_ips.get(0, os.environ.get("MASTER_ADDR", "localhost"))
                for attempt in range(max_attempts):
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(5)
                        sock.connect((root_addr, self.listen_port))
                        sock.settimeout(None)
                        sock.sendall(struct.pack(">I", self.rank))
                        self.peers["meta_root"] = sock
                        print(f"[dynamic-comm] Rank {self.rank} metadata channel to root ({root_addr})")
                        return
                    except (ConnectionRefusedError, OSError) as e:
                        if attempt % 5 == 0 and attempt > 0:
                            print(f"[dynamic-comm] Rank {self.rank} still waiting for root at {root_addr}:{self.listen_port} ({attempt * 2}s elapsed)...")
                        time.sleep(2)
                raise RuntimeError(
                    f"Could not connect metadata channel to root at {root_addr}:{self.listen_port} "
                    f"after {conn_timeout}s. Is the root node running?"
                )
            t = threading.Thread(target=connect_metadata, daemon=True)
            t.start()
            threads.append(t)

        # Wait for all connections
        for t in threads:
            t.join(timeout=conn_timeout + 10)

        print(f"[dynamic-comm] Rank {self.rank} all connections established ({len(self.peers)} peers)")

    def _connect_to_peers(self, peer_addresses):
        """Connect using explicit peer addresses (for hot-swap)."""
        for rank, addr in peer_addresses.items():
            host, port = addr.split(":")
            for attempt in range(30):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((host, int(port)))
                    sock.sendall(struct.pack(">I", self.rank))
                    self.peers[rank] = sock
                    print(f"[dynamic-comm] Connected to rank {rank} at {addr}")
                    break
                except ConnectionRefusedError:
                    time.sleep(2)

    # --- Forward send/recv (pipeline adjacent) ---

    def start_forward_recv(self):
        self.forward_ips = [
            torch.zeros(shape, dtype=self.dtype).to(self.device)
            for shape in self.forward_input_shapes
        ]
        prev_rank = self.rank - 1

        def do_recv():
            for i in range(len(self.forward_ips)):
                self.forward_ips[i] = TensorSocket.recv_tensor(
                    self.peers[prev_rank], device=str(self.device)
                )

        self.forward_recv_works = [AsyncWork(do_recv)]

    def trigger_forward_send(self, data):
        if self.forward_send_work is not None:
            self.forward_send_work.wait()
        next_rank = self.rank + 1
        self.forward_send_work = AsyncWork(
            TensorSocket.send_tensor, (self.peers[next_rank], data)
        )

    def forward_recv_works_done(self):
        if self.forward_recv_works is None:
            return False
        return all(w.is_completed() for w in self.forward_recv_works)

    # --- Backward send/recv (training only) ---

    def start_backward_recv(self):
        self.backward_ips = [
            torch.zeros(shape, dtype=self.dtype).to(self.device)
            for shape in self.backward_input_shapes
        ]
        next_rank = self.rank + 1

        def do_recv():
            for i in range(len(self.backward_ips)):
                self.backward_ips[i] = TensorSocket.recv_tensor(
                    self.peers[next_rank], device=str(self.device)
                )

        self.backward_recv_works = [AsyncWork(do_recv)]

    def trigger_backward_send(self, data):
        if self.backward_send_work is not None:
            self.backward_send_work.wait()
        prev_rank = self.rank - 1
        self.backward_send_work = AsyncWork(
            TensorSocket.send_tensor, (self.peers[prev_rank], data)
        )

    def backward_recv_works_done(self):
        if self.backward_recv_works is None:
            return False
        return all(w.is_completed() for w in self.backward_recv_works)

    # --- Feedback (leaf -> root token broadcast) ---

    def start_feedback_recv(self):
        self.feedback_ip = torch.zeros(self.feedback_shape, dtype=torch.int64).to(self.device)

        if self.node_type == NodeTypes.ROOT:
            # Root receives from leaf, then forwards to all stem nodes
            leaf_rank = self.world_size - 1
            meta_key = f"meta_{leaf_rank}"
            def recv_and_forward():
                token = TensorSocket.recv_tensor(self.peers[meta_key], device=str(self.device))
                self.feedback_ip = token
                # Forward to stem nodes (all meta_ peers except the leaf)
                for key, sock in self.peers.items():
                    if isinstance(key, str) and key.startswith("meta_") and key != meta_key:
                        TensorSocket.send_tensor(sock, token)
            self.feedback_recv_work = AsyncWork(recv_and_forward)
        else:
            # Non-root (stem/leaf) receives from root via metadata channel
            self.feedback_recv_work = AsyncWork(
                lambda: setattr(self, 'feedback_ip',
                    TensorSocket.recv_tensor(self.peers["meta_root"], device=str(self.device)))
            )

    def trigger_feedback_send(self, data):
        if self.feedback_send_work is not None:
            self.feedback_send_work.wait()

        if self.node_type == NodeTypes.LEAF:
            # Leaf sends to root, root forwards to all
            self.feedback_send_work = AsyncWork(
                TensorSocket.send_tensor, (self.peers["meta_root"], data)
            )
        elif self.node_type == NodeTypes.ROOT:
            # Root broadcasts to all non-root nodes
            def broadcast():
                for key, sock in self.peers.items():
                    if isinstance(key, str) and key.startswith("meta_"):
                        TensorSocket.send_tensor(sock, data)
            self.feedback_send_work = AsyncWork(broadcast)

    def feedback_recv_work_done(self):
        if self.feedback_recv_work is None:
            return False
        return self.feedback_recv_work.is_completed()

    # --- Metadata broadcast (root -> all) ---

    def broadcast_metadata(self, data):
        if self.node_type == NodeTypes.ROOT:
            # Send to all non-root nodes
            for key, sock in self.peers.items():
                if isinstance(key, str) and key.startswith("meta_"):
                    TensorSocket.send_tensor(sock, data)
        else:
            # Receive from root — block forever; user think-time between
            # requests is unbounded so a fixed timeout isn't appropriate here
            received = TensorSocket.recv_tensor(
                self.peers["meta_root"], device=str(self.device), timeout=None
            )
            data.copy_(received)

    def broadcast_metadata_objects(self, data):
        if self.node_type == NodeTypes.ROOT:
            for key, sock in self.peers.items():
                if isinstance(key, str) and key.startswith("meta_"):
                    TensorSocket.send_object(sock, data)
        else:
            # Block forever waiting for next prompt — user think-time is unbounded
            received = TensorSocket.recv_object(self.peers["meta_root"], timeout=None)
            for i in range(len(data)):
                if i < len(received):
                    data[i] = received[i]

    def gather_at_root(self, data, gather_list=None):
        if self.node_type == NodeTypes.ROOT:
            if gather_list is not None:
                gather_list[0] = data
                for key, sock in self.peers.items():
                    if isinstance(key, str) and key.startswith("meta_"):
                        rank = int(key.split("_")[1])
                        received = TensorSocket.recv_tensor(sock, device=str(self.device))
                        gather_list[rank] = received
        else:
            TensorSocket.send_tensor(self.peers["meta_root"], data)

    # --- Dynamic node management ---

    def add_node(self, rank, address):
        """Hot-add a new node to the pipeline."""
        host, port = address.split(":")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, int(port)))
        sock.sendall(struct.pack(">I", self.rank))
        with self._lock:
            self.peers[rank] = sock
            self.world_size += 1
        print(f"[dynamic-comm] Hot-added rank {rank} at {address}")

    def remove_node(self, rank):
        """Hot-remove a node from the pipeline."""
        with self._lock:
            if rank in self.peers:
                try:
                    self.peers[rank].close()
                except Exception:
                    pass
                del self.peers[rank]
                self.world_size -= 1
        print(f"[dynamic-comm] Removed rank {rank}")

    def close(self):
        """Close all connections."""
        for sock in self.peers.values():
            try:
                sock.close()
            except Exception:
                pass
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
        self.peers.clear()
