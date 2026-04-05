"""
Hardware profiling and auto-proportioning for heterogeneous clusters.

Before torch.distributed init, the root node collects hardware specs from all
nodes via a simple TCP socket, computes optimal layer proportions based on
available memory, and sends the proportions back to each node.
"""

import json
import os
import socket
import subprocess
import time


def get_hardware_info():
    """Profile this machine's hardware for inference capacity."""
    info = {"type": "cpu", "memory_gb": 0, "name": "CPU"}

    # Try GPU first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(",")
            info["type"] = "cuda"
            info["name"] = parts[0].strip()
            info["memory_gb"] = round(int(parts[1].strip()) / 1024, 1)
            return info
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: try torch
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["type"] = "cuda"
            info["name"] = props.name
            info["memory_gb"] = round(props.total_mem / (1024**3), 1)
            return info
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple Silicon: unified memory, report system RAM as proxy
            info["type"] = "mps"
            info["name"] = "Apple Silicon GPU"
            try:
                import psutil
                info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
            except ImportError:
                info["memory_gb"] = 8.0
            return info
    except ImportError:
        pass

    # CPU: use available RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["memory_gb"] = round(mem.available / (1024**3), 1)
        info["name"] = "CPU"
    except ImportError:
        info["memory_gb"] = 4.0  # conservative default

    return info


def compute_proportions(hardware_list):
    """Compute layer proportions from a list of hardware info dicts.

    Proportional to available memory. GPU memory counts 10x vs CPU RAM
    because GPU inference is ~10x faster per layer.
    """
    weights = []
    for hw in hardware_list:
        mem = hw["memory_gb"]
        if hw["type"] == "cuda":
            weights.append(mem * 10)  # discrete GPU ~10x CPU
        elif hw["type"] == "mps":
            weights.append(mem * 4)   # Apple Silicon GPU ~4x CPU (unified memory)
        else:
            weights.append(mem)

    total = sum(weights)
    if total == 0:
        # Equal split fallback
        n = len(hardware_list)
        return [1.0 / n] * n

    proportions = [w / total for w in weights]

    # Round to 2 decimal places, ensure sum = 1.0
    proportions = [round(p, 2) for p in proportions]
    proportions[-1] = round(1.0 - sum(proportions[:-1]), 2)

    return proportions


def collect_hardware_as_root(world_size, port=29400):
    """Root node: listen for hardware info from all other nodes.

    Returns list of hardware info dicts, ordered by rank.
    """
    my_info = get_hardware_info()
    hardware_list = [None] * world_size
    hardware_list[0] = my_info

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(world_size - 1)
    server.settimeout(300)  # 5 min timeout for all nodes to connect

    print(f"[auto-profile] Root listening on port {port}, waiting for {world_size - 1} nodes...")

    for _ in range(world_size - 1):
        conn, addr = server.accept()
        data = conn.recv(4096).decode()
        msg = json.loads(data)
        rank = msg["rank"]
        hw = msg["hardware"]
        hardware_list[rank] = hw
        print(f"[auto-profile] Node {rank} ({addr[0]}): {hw['name']} {hw['memory_gb']}GB {hw['type']}")

        # Don't send proportions yet, wait for all nodes
        conn.close()

    server.close()

    # Compute proportions
    proportions = compute_proportions(hardware_list)
    print(f"[auto-profile] Hardware: {[h['name'] + ' ' + str(h['memory_gb']) + 'GB' for h in hardware_list]}")
    print(f"[auto-profile] Proportions: {proportions}")

    # Send proportions to all nodes via a second round
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(world_size - 1)
    server.settimeout(300)

    for _ in range(world_size - 1):
        conn, addr = server.accept()
        conn.sendall(json.dumps({"proportions": proportions}).encode())
        conn.close()

    server.close()
    return hardware_list, proportions


def report_hardware_to_root(rank, master_addr, port=29400):
    """Non-root node: send hardware info to root and receive proportions."""
    my_info = get_hardware_info()
    msg = json.dumps({"rank": rank, "hardware": my_info})

    # Retry connection (root might not be listening yet)
    for attempt in range(30):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((master_addr, port))
            sock.sendall(msg.encode())
            sock.close()
            break
        except ConnectionRefusedError:
            print(f"[auto-profile] Waiting for root ({master_addr}:{port})... attempt {attempt + 1}")
            time.sleep(2)
    else:
        raise RuntimeError(f"Could not connect to root at {master_addr}:{port}")

    # Second round: receive proportions
    for attempt in range(30):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((master_addr, port))
            data = sock.recv(4096).decode()
            sock.close()
            result = json.loads(data)
            return my_info, result["proportions"]
        except ConnectionRefusedError:
            time.sleep(1)

    raise RuntimeError("Could not receive proportions from root")
