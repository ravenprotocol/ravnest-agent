"""
Ravnest Worker

Registers with a coordinator, receives cluster config (rank, proportions),
and runs the inference pipeline in-process. Monitors for cluster changes
and hot-reconfigures: rebuilds TCP connections and re-splits layers without
restarting the process. The model stays in memory, only layers are re-assigned.

Usage:
    ravnest worker --coordinator http://master:8080
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import threading
import requests
import torch

from .hardware import get_hardware_info


def get_node_id():
    """Get a routable IP for this node."""
    # Tailscale IP (best for cross-network)
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # LAN IP (works in Docker networks and LANs)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        pass
    return socket.gethostname()


class Worker:
    def __init__(self, coordinator_url, model_name=None, device=None):
        self.coordinator_url = coordinator_url.rstrip("/")
        self.node_id = get_node_id()
        self.hardware = get_hardware_info()
        self.model_name = model_name
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_str)
        self.current_version = -1
        self.running = True

        # In-process inference state
        self.full_model = None       # original model (not pruned)
        self.tokenizer = None
        self.node = None             # current Node instance
        self.engine = None           # current InferenceEngine
        self.api_app = None          # FastAPI app
        self.api_thread = None       # uvicorn thread
        self.inference_thread = None # leaf receive loop thread
        self.reconfiguring = threading.Event()
        self.reconfiguring.set()     # starts as "not reconfiguring"

    def register(self):
        for attempt in range(60):
            try:
                resp = requests.post(
                    f"{self.coordinator_url}/register",
                    json={"node_id": self.node_id, "hardware": self.hardware},
                    timeout=10,
                )
                resp.raise_for_status()
                status = resp.json()
                print(f"[worker] Registered as {self.node_id}")
                print(f"[worker] Cluster: {status['num_nodes']}/{status['min_nodes']} nodes, "
                      f"ready={status['ready']}")
                return status
            except requests.ConnectionError:
                print(f"[worker] Waiting for coordinator... attempt {attempt + 1}")
                time.sleep(5)
            except Exception as e:
                print(f"[worker] Registration error: {e}")
                time.sleep(5)
        raise RuntimeError(f"Could not register with coordinator at {self.coordinator_url}")

    def get_config(self):
        try:
            resp = requests.get(f"{self.coordinator_url}/config/{self.node_id}", timeout=10)
            if resp.status_code == 503:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[worker] Config error: {e}")
            return None

    def heartbeat(self):
        try:
            resp = requests.post(f"{self.coordinator_url}/heartbeat/{self.node_id}", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[worker] Heartbeat error: {e}")
            return None

    def leave(self):
        try:
            requests.post(f"{self.coordinator_url}/leave/{self.node_id}", timeout=5)
        except Exception:
            pass

    def load_model(self, model_name):
        """Load model and tokenizer once. Kept in memory across reconfigurations."""
        if self.full_model is not None and self.model_name == model_name:
            print("[worker] Model already loaded, reusing")
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ravnest.lazy_init.lazy_context import LazyInitContext

        self.model_name = model_name
        cache_dir = os.environ.get("MODEL_CACHE", "/tmp/ravnest_model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        use_cpu = self.device_str == "cpu"

        print(f"[worker] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        print(f"[worker] Loading model: {model_name}")
        dtype = torch.float32 if use_cpu else torch.float16
        init_ctx = LazyInitContext()
        with init_ctx:
            self.full_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype,
                device_map="cpu" if use_cpu else "cuda",
                cache_dir=cache_dir,
            )
        self.full_model.eval()
        print(f"[worker] Model loaded: {model_name}")

    def configure_pipeline(self, config):
        """Create Node + InferenceEngine with the given config.

        Uses the dynamic backend so connections can be rebuilt
        without restarting the process.
        """
        import copy
        from ravnest.node_tcp import Node
        from ravnest.inference.inference_engine import InferenceEngine

        rank = config["rank"]
        world_size = config["world_size"]
        proportions = config["proportions"]
        master_addr = config["master_addr"]
        # peer_ips: {rank_int: ip_string} from coordinator
        peers_raw = config.get("peers", {})
        peer_ips = {int(k): v for k, v in peers_raw.items()}

        # Close old communication if exists
        if self.node and hasattr(self.node, 'comm_session'):
            if hasattr(self.node.comm_session, 'close'):
                self.node.comm_session.close()

        # Set env vars for the dynamic backend
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = "29500"

        use_cpu = self.device_str == "cpu"

        # Deep copy the model so we can re-split from fresh each time
        print(f"[worker] Copying model for rank {rank} (proportions: {proportions})...")
        model_copy = copy.deepcopy(self.full_model)
        model_copy.eval()

        print(f"[worker] Creating Node (rank={rank}, world_size={world_size}, backend=dynamic, peers={peer_ips})...")
        self.node = Node(
            model=model_copy,
            device=self.device,
            dtype="float32" if use_cpu else "float16",
            batch_size=1,
            mode="inference",
            seq_length=5,
            backend="dynamic",
            cluster_length=world_size,
            reduce_factor=1,
            proportions=proportions,
            peer_ips=peer_ips,
        )
        self.node.model.eval()

        print(f"[worker] Creating InferenceEngine...")
        self.engine = InferenceEngine(self.node, self.tokenizer)

        print(f"[worker] Pipeline ready. Layers {self.node.layer_start_idx}-{self.node.layer_end_idx}")
        return self.engine

    def start_api_server(self, port=8000):
        """Start FastAPI server in a background thread (root only)."""
        if self.api_thread and self.api_thread.is_alive():
            return  # already running

        from deploy.api_server import create_app
        import uvicorn

        # Create app with a wrapper that checks reconfiguring state
        self.api_app = create_app(self.engine, self.tokenizer)

        def run_server():
            uvicorn.run(self.api_app, host="0.0.0.0", port=port, log_level="info")

        self.api_thread = threading.Thread(target=run_server, daemon=True)
        self.api_thread.start()
        print(f"[worker] API server started on port {port}")

    def start_leaf_loop(self):
        """Start leaf receive loop in a background thread."""
        if self.inference_thread and self.inference_thread.is_alive():
            return

        def leaf_loop():
            print("[worker] Leaf receive loop started")
            while self.running:
                if not self.reconfiguring.is_set():
                    time.sleep(0.5)
                    continue
                try:
                    self.engine.generate(prompt_list=None, max_seq_lengths=None)
                except RuntimeError as e:
                    if not self.running:
                        break
                    print(f"[worker] Generation error: {e}")
                    time.sleep(5)
                except Exception as e:
                    if not self.running:
                        break
                    print(f"[worker] Unexpected error: {e}")
                    time.sleep(5)

        self.inference_thread = threading.Thread(target=leaf_loop, daemon=True)
        self.inference_thread.start()

    def reconfigure(self, config):
        """Hot-reconfigure: rebuild connections and re-split layers.

        The model stays in memory. Only communication and layer assignment change.
        API requests get 503 during reconfiguration (~10-30s).
        """
        print(f"[worker] === RECONFIGURING (v{self.current_version} -> v{config['cluster_version']}) ===")
        self.reconfiguring.clear()  # signal "reconfiguring"

        try:
            # Load model if not loaded yet
            model_name = config.get("model") or self.model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.load_model(model_name)

            # Build new pipeline
            self.configure_pipeline(config)

            # Start/restart role-specific loops
            rank = config["rank"]
            if rank == 0:
                # Update the API server's engine reference
                if self.api_app:
                    # The API server holds a reference to engine via closure.
                    # We need to restart it with the new engine.
                    pass
                self.start_api_server()
            else:
                self.start_leaf_loop()

            self.current_version = config["cluster_version"]
            print(f"[worker] === RECONFIGURED (v{self.current_version}) ===")

        finally:
            self.reconfiguring.set()  # signal "ready"

    def run(self):
        """Main worker loop."""
        print(f"[worker] Node ID: {self.node_id}")
        print(f"[worker] Hardware: {self.hardware['name']} "
              f"{self.hardware['memory_gb']}GB {self.hardware['type']}")
        print(f"[worker] Coordinator: {self.coordinator_url}")
        print(f"[worker] Device: {self.device_str}")

        def shutdown(sig, frame):
            print("\n[worker] Shutting down...")
            self.running = False
            if self.node and hasattr(self.node, 'comm_session') and hasattr(self.node.comm_session, 'close'):
                self.node.comm_session.close()
            self.leave()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        self.register()

        while self.running:
            config = self.get_config()

            if config is None:
                print("[worker] Waiting for cluster to be ready...")
                time.sleep(5)
                self.heartbeat()
                continue

            if config["cluster_version"] != self.current_version:
                self.reconfigure(config)

            status = self.heartbeat()
            if status and status["cluster_version"] != self.current_version:
                config = self.get_config()
                if config:
                    self.reconfigure(config)

            time.sleep(10)
