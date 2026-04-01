"""
Ravnest Worker

Registers with a coordinator, receives cluster config (rank, proportions),
and runs the inference pipeline. Monitors for cluster changes and restarts
when the cluster topology changes.

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

from .hardware import get_hardware_info


def get_node_id():
    """Get a stable identifier for this node."""
    # Use Tailscale IP if available
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fall back to hostname
    return socket.gethostname()


class Worker:
    def __init__(self, coordinator_url, model=None, device=None):
        self.coordinator_url = coordinator_url.rstrip("/")
        self.node_id = get_node_id()
        self.hardware = get_hardware_info()
        self.model = model
        self.device = device
        self.current_version = -1
        self.inference_process = None
        self.running = True

    def register(self):
        """Register with the coordinator and get cluster status."""
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
                print(f"[worker] Waiting for coordinator ({self.coordinator_url})... "
                      f"attempt {attempt + 1}")
                time.sleep(5)
            except Exception as e:
                print(f"[worker] Registration error: {e}")
                time.sleep(5)

        raise RuntimeError(f"Could not register with coordinator at {self.coordinator_url}")

    def get_config(self):
        """Get inference config from coordinator."""
        try:
            resp = requests.get(
                f"{self.coordinator_url}/config/{self.node_id}",
                timeout=10,
            )
            if resp.status_code == 503:
                return None  # cluster not ready
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[worker] Config error: {e}")
            return None

    def heartbeat(self):
        """Send heartbeat to coordinator, returns cluster status."""
        try:
            resp = requests.post(
                f"{self.coordinator_url}/heartbeat/{self.node_id}",
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[worker] Heartbeat error: {e}")
            return None

    def leave(self):
        """Notify coordinator that we're leaving."""
        try:
            requests.post(
                f"{self.coordinator_url}/leave/{self.node_id}",
                timeout=5,
            )
        except Exception:
            pass

    def start_inference(self, config):
        """Start the inference process with the given config."""
        self.stop_inference()

        env = os.environ.copy()
        env["RANK"] = str(config["rank"])
        env["WORLD_SIZE"] = str(config["world_size"])
        env["MASTER_ADDR"] = config["master_addr"]
        env["MASTER_PORT"] = "29500"
        env["MODEL_NAME"] = config.get("model") or self.model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        env["NODE_ROLE"] = "root" if config["rank"] == 0 else "leaf"
        env["RAVNEST_DEVICE"] = config.get("device") or self.device or "cpu"
        env["RAVNEST_PROPORTIONS"] = ",".join(str(p) for p in config["proportions"])
        env["PYTHONUNBUFFERED"] = "1"

        # Find entrypoint
        deploy_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deploy")
        entrypoint = os.path.join(deploy_dir, "entrypoint.py")

        if not os.path.exists(entrypoint):
            print(f"[worker] Error: entrypoint not found at {entrypoint}")
            return

        print(f"[worker] Starting inference (rank={config['rank']}, "
              f"world_size={config['world_size']}, "
              f"proportions={config['proportions']})")

        self.inference_process = subprocess.Popen(
            [sys.executable, entrypoint],
            env=env,
        )
        self.current_version = config["cluster_version"]

    def stop_inference(self):
        """Stop the current inference process."""
        if self.inference_process is not None:
            print("[worker] Stopping inference process...")
            self.inference_process.terminate()
            try:
                self.inference_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.inference_process.kill()
            self.inference_process = None

    def run(self):
        """Main worker loop: register, start inference, monitor for changes."""
        print(f"[worker] Node ID: {self.node_id}")
        print(f"[worker] Hardware: {self.hardware['name']} "
              f"{self.hardware['memory_gb']}GB {self.hardware['type']}")
        print(f"[worker] Coordinator: {self.coordinator_url}")

        # Handle graceful shutdown
        def shutdown(sig, frame):
            print("\n[worker] Shutting down...")
            self.running = False
            self.stop_inference()
            self.leave()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # Register
        self.register()

        # Main loop: wait for cluster ready, start inference, monitor
        while self.running:
            config = self.get_config()

            if config is None:
                # Cluster not ready
                print("[worker] Waiting for cluster to be ready...")
                time.sleep(5)
                self.heartbeat()
                continue

            if config["cluster_version"] != self.current_version:
                # Cluster changed, (re)start inference
                print(f"[worker] Cluster version changed "
                      f"({self.current_version} -> {config['cluster_version']})")
                self.start_inference(config)

            # Check if inference process is still alive
            if self.inference_process and self.inference_process.poll() is not None:
                print(f"[worker] Inference process exited with code "
                      f"{self.inference_process.returncode}")
                self.inference_process = None

            # Heartbeat and check for cluster changes
            status = self.heartbeat()
            if status and status["cluster_version"] != self.current_version:
                print(f"[worker] Cluster topology changed, restarting...")
                config = self.get_config()
                if config:
                    self.start_inference(config)

            time.sleep(10)
