"""
ravnest CLI — one-command distributed inference.

Usage:
    ravnest up [--model MODEL] [--nodes N] [--device cpu|cuda] [--port PORT] [--api-key KEY]
    ravnest up --master-addr IP [--model MODEL] [--nodes N]   # cross-machine root
    ravnest join --master-addr IP [--model MODEL] [--rank R] [--world-size N]
    ravnest down
    ravnest status
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import textwrap


def detect_hardware():
    """Detect available GPUs and return device info."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = line.split(",")
                name = parts[0].strip()
                vram_mb = int(parts[1].strip())
                gpus.append({"name": name, "vram_gb": round(vram_mb / 1024, 1)})
            return {"device": "cuda", "gpu_count": len(gpus), "gpus": gpus}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpus = []
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "name": props.name,
                    "vram_gb": round(props.total_mem / (1024**3), 1),
                })
            return {"device": "cuda", "gpu_count": gpu_count, "gpus": gpus}
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                import psutil
                mem_gb = round(psutil.virtual_memory().total / (1024**3), 1)
            except ImportError:
                mem_gb = 8.0
            return {
                "device": "mps",
                "gpu_count": 1,
                "gpus": [{"name": "Apple Silicon GPU", "vram_gb": mem_gb}],
            }
    except ImportError:
        pass

    return {"device": "cpu", "gpu_count": 0, "gpus": []}


def get_local_ip():
    """Get this machine's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def generate_compose(model, nodes, device, port, api_key="", master_addr="node-0",
                     rank_offset=0, world_size=None, network_mode=None, proportions=None,
                     auto_profile=False):
    """Generate a docker-compose.yml string."""
    if world_size is None:
        world_size = nodes

    if device == "cuda":
        dockerfile = "deploy/Dockerfile"
        gpu_block = textwrap.dedent("""\
        runtime: nvidia
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: 1
                  capabilities: [gpu]""")
    else:
        dockerfile = "deploy/Dockerfile.cpu"
        gpu_block = ""

    services = []
    for i in range(nodes):
        rank = rank_offset + i
        role = "root" if rank == 0 else "leaf"
        name = f"node-{rank}"

        env_lines = [
            f"      - RANK={rank}",
            f"      - WORLD_SIZE={world_size}",
            f"      - MASTER_ADDR={master_addr}",
            f"      - MASTER_PORT=29500",
            f"      - MODEL_NAME={model}",
            f"      - NODE_ROLE={role}",
            f"      - RAVNEST_DEVICE={device}",
            f"      - RAVNEST_API_KEY={api_key}" if api_key and rank == 0 else None,
            f"      - RAVNEST_PROPORTIONS={','.join(str(p) for p in proportions)}" if proportions else None,
            f"      - RAVNEST_AUTO_PROFILE=true" if auto_profile else None,
            f"      - PYTHONUNBUFFERED=1",
        ]
        env_lines = [line for line in env_lines if line is not None]

        svc = f"""  {name}:
    build:
      context: ..
      dockerfile: {dockerfile}
    environment:
{chr(10).join(env_lines)}
    volumes:
      - model_cache:/app/model_cache
    command: python deploy/entrypoint.py"""

        if network_mode:
            svc += f"\n    network_mode: {network_mode}"

        if rank == 0 and not network_mode:
            svc += f"""
    ports:
      - "{port}:8000\""""

        if i > 0 and not network_mode:
            first_name = f"node-{rank_offset}"
            svc += f"""
    depends_on:
      - {first_name}"""

        if gpu_block:
            svc += f"\n    {gpu_block}"

        services.append(svc)

    compose = f"""services:
{chr(10).join(services)}

volumes:
  model_cache:
"""
    return compose


def find_deploy_dir():
    """Find the deploy directory relative to this package."""
    repo_deploy = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deploy")
    if os.path.isdir(repo_deploy):
        return repo_deploy
    if os.path.isdir("deploy"):
        return os.path.abspath("deploy")
    return None


def cmd_up(args):
    hw = detect_hardware()

    device = args.device
    if device is None:
        device = hw["device"]

    model = args.model
    if model is None:
        if device == "cuda":
            model = "meta-llama/Llama-3.2-3B"
        else:
            model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    nodes = args.nodes
    port = args.port
    api_key = args.api_key or ""
    master_addr = args.master_addr
    world_size = args.world_size or nodes
    proportions = [float(p) for p in args.proportions.split(",")] if args.proportions else None
    auto_profile = args.auto_profile or (master_addr is not None and proportions is None)

    # Cross-machine mode: use host networking so Gloo can reach other machines
    cross_machine = master_addr is not None
    if cross_machine:
        network_mode = "host"
    else:
        master_addr = "node-0"
        network_mode = None

    local_ip = get_local_ip()

    print(f"Ravnest Distributed Inference")
    print(f"  Device:  {device}")
    if device == "cuda" and hw["gpus"]:
        for i, gpu in enumerate(hw["gpus"]):
            print(f"  GPU {i}:   {gpu['name']} ({gpu['vram_gb']} GB)")
    print(f"  Model:   {model}")
    print(f"  Nodes:   {nodes} (on this machine)")
    if cross_machine:
        print(f"  Mode:    cross-machine (master: {master_addr})")
        print(f"  This IP: {local_ip}")
    else:
        print(f"  Mode:    single-machine")
    print(f"  API:     http://{'0.0.0.0' if cross_machine else 'localhost'}:{port}")
    print(f"  Auth:    {'enabled' if api_key else 'disabled'}")
    if proportions:
        print(f"  Split:   {proportions}")
    elif auto_profile:
        print(f"  Split:   auto-profile (detecting hardware)")
    print()

    if cross_machine:
        print(f"To add more machines, run on each one:")
        print(f"  ravnest join --master-addr {master_addr} --model {model} --world-size {world_size}")
        print()

    deploy_dir = find_deploy_dir()
    if deploy_dir is None:
        print("Error: cannot find deploy/ directory. Run from the ravnest repo root.")
        sys.exit(1)

    compose_content = generate_compose(
        model, nodes, device, port, api_key,
        master_addr=master_addr,
        world_size=world_size,
        network_mode=network_mode,
        proportions=proportions,
        auto_profile=auto_profile,
    )
    compose_path = os.path.join(deploy_dir, "docker-compose.generated.yml")

    with open(compose_path, "w") as f:
        f.write(compose_content)

    print(f"Generated {compose_path}")
    print("Starting containers...")
    print()

    os.chdir(deploy_dir)
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.generated.yml", "up", "--build"],
        check=False,
    )


def cmd_join(args):
    """Join an existing cluster as a leaf node on a different machine."""
    hw = detect_hardware()

    device = args.device
    if device is None:
        device = hw["device"]

    model = args.model
    if model is None:
        if device == "cuda":
            model = "meta-llama/Llama-3.2-3B"
        else:
            model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    master_addr = args.master_addr
    rank = args.rank
    world_size = args.world_size
    proportions = [float(p) for p in args.proportions.split(",")] if args.proportions else None
    auto_profile = args.auto_profile or (proportions is None)
    local_ip = get_local_ip()

    print(f"Ravnest Distributed Inference — Joining Cluster")
    print(f"  Device:      {device}")
    if device == "cuda" and hw["gpus"]:
        for i, gpu in enumerate(hw["gpus"]):
            print(f"  GPU {i}:       {gpu['name']} ({gpu['vram_gb']} GB)")
    print(f"  Model:       {model}")
    print(f"  Master:      {master_addr}")
    print(f"  This IP:     {local_ip}")
    print(f"  Rank:        {rank}")
    print(f"  World size:  {world_size}")
    print()

    deploy_dir = find_deploy_dir()
    if deploy_dir is None:
        print("Error: cannot find deploy/ directory. Run from the ravnest repo root.")
        sys.exit(1)

    compose_content = generate_compose(
        model,
        nodes=1,
        device=device,
        port=8000,
        master_addr=master_addr,
        rank_offset=rank,
        world_size=world_size,
        network_mode="host",
        proportions=proportions,
        auto_profile=auto_profile,
    )
    compose_path = os.path.join(deploy_dir, "docker-compose.generated.yml")

    with open(compose_path, "w") as f:
        f.write(compose_content)

    print(f"Generated {compose_path}")
    print("Starting container (waiting for master to be ready)...")
    print()

    os.chdir(deploy_dir)
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.generated.yml", "up", "--build"],
        check=False,
    )


def cmd_down(args):
    deploy_dir = find_deploy_dir()
    if deploy_dir is None:
        print("Error: cannot find deploy/ directory.")
        sys.exit(1)

    compose_path = os.path.join(deploy_dir, "docker-compose.generated.yml")
    if not os.path.exists(compose_path):
        print("No running ravnest instance found.")
        sys.exit(1)

    os.chdir(deploy_dir)
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.generated.yml", "down"],
        check=False,
    )


def cmd_status(args):
    deploy_dir = find_deploy_dir()
    if deploy_dir is None:
        print("No deploy directory found.")
        return

    compose_path = os.path.join(deploy_dir, "docker-compose.generated.yml")
    if not os.path.exists(compose_path):
        print("No running ravnest instance found.")
        return

    os.chdir(deploy_dir)
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.generated.yml", "ps"],
        check=False,
    )


def cmd_coordinator(args):
    """Run the cluster coordinator."""
    import importlib.util
    # Direct import to avoid triggering ravnest.__init__ (which needs torch)
    cli_dir = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("coordinator", os.path.join(cli_dir, "coordinator.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    create_coordinator_app = mod.create_coordinator_app
    import uvicorn

    local_ip = get_local_ip()
    print(f"Ravnest Cluster Coordinator")
    print(f"  This IP:    {local_ip}")
    print(f"  Port:       {args.port}")
    print(f"  Min nodes:  {args.min_nodes}")
    print(f"  Heartbeat:  {args.heartbeat_timeout}s timeout")
    if args.model:
        print(f"  Model:      {args.model}")
    print()
    print(f"Workers connect with:")
    print(f"  ravnest worker --coordinator http://{local_ip}:{args.port}")
    print()

    app = create_coordinator_app(
        min_nodes=args.min_nodes,
        heartbeat_timeout=args.heartbeat_timeout,
        model=args.model,
        device=args.device,
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


def _find_free_port(start_port):
    """Find a free TCP port, starting from start_port."""
    for port in range(start_port, start_port + 100):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", port))
            s.close()
            return port
        except OSError:
            continue
    return start_port  # fallback, let it fail later with a clear error


def _pre_download_model(model, cache_dir):
    """Download model and tokenizer once before spawning nodes."""
    print(f"Downloading model: {model}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  (subsequent runs will use the cached copy)")
    print()
    subprocess.run(
        [sys.executable, "-c", f"""
import os
os.environ['HF_HOME'] = '{cache_dir}'
os.environ['TRANSFORMERS_CACHE'] = '{cache_dir}'
from transformers import AutoTokenizer, AutoModelForCausalLM
print('  Downloading tokenizer...')
AutoTokenizer.from_pretrained('{model}', cache_dir='{cache_dir}')
print('  Downloading model weights...')
AutoModelForCausalLM.from_pretrained('{model}', cache_dir='{cache_dir}')
print('  Done!')
"""],
        check=True,
    )
    print()


def cmd_native(args):
    """Launch a cluster as native processes (no Docker).

    Primarily for platforms where Docker can't access the local GPU
    (macOS / Apple Silicon MPS), and for multi-machine runs over
    Tailscale or LAN.
    """
    import time as _time

    hw = detect_hardware()
    device = args.device or hw["device"]
    model = args.model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    port = args.port

    # Find entrypoint.py: either in-repo (deploy/entrypoint.py) or shipped with package
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    entrypoint = os.path.join(repo_root, "deploy", "entrypoint.py")
    if not os.path.exists(entrypoint):
        print(f"Error: cannot find deploy/entrypoint.py. Run from a ravnest checkout.")
        sys.exit(1)

    # Multi-machine mode: user passes --peers host1,host2,... and --rank R
    # Single-machine mode: default, runs all ranks on 127.0.0.1
    if args.peers:
        peer_list = [h.strip() for h in args.peers.split(",")]
        world_size = args.world_size or len(peer_list)
        if len(peer_list) != world_size:
            print(f"Error: --peers has {len(peer_list)} entries but --world-size is {world_size}")
            sys.exit(1)
        if args.rank is None:
            print("Error: --peers requires --rank (which node am I in the peer list?)")
            sys.exit(1)
        ranks_to_run = [args.rank]
    else:
        world_size = args.world_size or args.nodes
        peer_list = ["127.0.0.1"] * world_size
        ranks_to_run = list(range(world_size))

    peers = ",".join(peer_list)

    # Auto-find a free port if the requested one is taken (root only)
    if 0 in ranks_to_run:
        actual_port = _find_free_port(port)
        if actual_port != port:
            print(f"Port {port} is in use, using {actual_port} instead.")
        port = actual_port

    # Auto-profile: exchange hardware info and compute proportions
    proportions_str = ""
    if args.auto_profile and args.peers:
        print(f"Auto-profiling hardware...")
        from ravnest.hardware import get_hardware_info, compute_proportions
        my_hw = get_hardware_info()
        print(f"  This node: {my_hw['name']} ({my_hw['type']}, {my_hw['memory_gb']}GB)")
        # For multi-machine: user must pass same --auto-profile on all nodes.
        # Each node sends its hw info to root, root computes proportions.
        # For now, estimate: MPS/CUDA nodes get 4x/10x weight vs CPU.
        # We compute locally with a placeholder for the remote node.
        if args.rank == 0:
            print(f"  Root node will collect hardware from {world_size - 1} peer(s)...")
            from ravnest.hardware import collect_hardware_as_root
            hw_list, proportions = collect_hardware_as_root(world_size)
            proportions_str = ",".join(str(p) for p in proportions)
            print(f"  Proportions: {proportions}")
        else:
            print(f"  Reporting hardware to root...")
            from ravnest.hardware import report_hardware_to_root
            _, proportions = report_hardware_to_root(args.rank, peer_list[0])
            proportions_str = ",".join(str(p) for p in proportions)
            print(f"  Proportions: {proportions}")
        print()

    print(f"Ravnest Native (no Docker)")
    print(f"  Model:       {model}")
    print(f"  Device:      {device}")
    print(f"  World size:  {world_size}")
    print(f"  Peers:       {peers}")
    print(f"  Running:     rank(s) {ranks_to_run}")
    if 0 in ranks_to_run:
        print(f"  API:         http://localhost:{port}")
    print()

    # Pre-download model before spawning nodes (avoids parallel downloads + race)
    cache_dir = os.environ.get("MODEL_CACHE_DIR",
                               os.path.expanduser("~/.cache/ravnest/models"))
    os.makedirs(cache_dir, exist_ok=True)
    # Check if model is already cached (look for config.json in HF cache structure)
    model_cached = any(
        os.path.exists(os.path.join(cache_dir, d, "config.json"))
        for d in os.listdir(cache_dir)
        if d.startswith("models--")
    ) if os.path.isdir(cache_dir) and os.listdir(cache_dir) else False
    if not model_cached:
        _pre_download_model(model, cache_dir)

    # Spawn the requested rank(s) as subprocess(es)
    procs = []
    try:
        for rank in ranks_to_run:
            is_root = rank == 0
            env = os.environ.copy()
            env.update({
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": peer_list[0],
                "MASTER_PORT": "29500",
                "MODEL_NAME": model,
                "NODE_ROLE": "root" if is_root else "leaf",
                "RAVNEST_DEVICE": device,
                "RAVNEST_BACKEND": "dynamic",
                "RAVNEST_PEERS": peers,
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": repo_root + os.pathsep + env.get("PYTHONPATH", ""),
                "MODEL_CACHE_DIR": cache_dir,
            })
            # MPS fallback: auto-route unsupported ops to CPU
            if device == "mps":
                env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            if is_root:
                env["RAVNEST_API_PORT"] = str(port)
            if args.api_key:
                env["RAVNEST_API_KEY"] = args.api_key
            if proportions_str:
                env["RAVNEST_PROPORTIONS"] = proportions_str

            role_str = "root+API" if is_root else "leaf"
            print(f"Starting node-{rank} ({role_str})...")
            p = subprocess.Popen([sys.executable, entrypoint], env=env)
            procs.append(p)

        # Wait for any process to exit
        while True:
            for p in procs:
                if p.poll() is not None:
                    print(f"\nnode exited with code {p.returncode}, shutting down cluster")
                    raise KeyboardInterrupt
            _time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping nodes...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


def cmd_worker(args):
    """Run as a worker managed by a coordinator."""
    try:
        from .worker import Worker
    except ImportError:
        from ravnest.worker import Worker

    hw = detect_hardware()
    device = args.device or hw["device"]

    worker = Worker(
        coordinator_url=args.coordinator,
        model=args.model,
        device=device,
    )
    worker.run()


SUPPORTED_MODELS = [
    {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "arch": "Llama", "params": "1.1B", "size": "~2GB", "min_ram": "4GB", "gated": False, "note": "Good for CPU testing"},
    {"id": "meta-llama/Llama-3.2-1B", "arch": "Llama", "params": "1B", "size": "~2GB", "min_ram": "4GB", "gated": True, "note": ""},
    {"id": "meta-llama/Llama-3.2-3B", "arch": "Llama", "params": "3B", "size": "~6GB", "min_ram": "8GB", "gated": True, "note": "Default for GPU"},
    {"id": "meta-llama/Llama-3.1-8B", "arch": "Llama", "params": "8B", "size": "~16GB", "min_ram": "20GB", "gated": True, "note": ""},
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "arch": "Llama", "params": "8B", "size": "~16GB", "min_ram": "20GB", "gated": True, "note": "Chat-tuned"},
    {"id": "mistralai/Mistral-7B-v0.1", "arch": "Mistral", "params": "7B", "size": "~14GB", "min_ram": "18GB", "gated": False, "note": ""},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "arch": "Mistral", "params": "7B", "size": "~14GB", "min_ram": "18GB", "gated": False, "note": "Chat-tuned"},
    {"id": "microsoft/Phi-3-mini-4k-instruct", "arch": "Phi-3", "params": "3.8B", "size": "~8GB", "min_ram": "10GB", "gated": False, "note": ""},
    {"id": "microsoft/phi-2", "arch": "Phi", "params": "2.7B", "size": "~6GB", "min_ram": "8GB", "gated": False, "note": ""},
    {"id": "Qwen/Qwen2-1.5B", "arch": "Qwen-2", "params": "1.5B", "size": "~3GB", "min_ram": "6GB", "gated": False, "note": ""},
    {"id": "Qwen/Qwen2-7B", "arch": "Qwen-2", "params": "7B", "size": "~14GB", "min_ram": "18GB", "gated": False, "note": ""},
]


def cmd_models(args):
    """List supported models."""
    print("Supported Models for Ravnest Distributed Inference")
    print("=" * 90)
    print(f"{'Model':<45} {'Arch':<10} {'Params':<8} {'Size':<8} {'Min RAM':<8} {'Note'}")
    print("-" * 90)
    for m in SUPPORTED_MODELS:
        gated = " [gated]" if m["gated"] else ""
        note = m["note"] + gated
        print(f"{m['id']:<45} {m['arch']:<10} {m['params']:<8} {m['size']:<8} {m['min_ram']:<8} {note}")
    print()
    print("Gated models require a HuggingFace account + access request.")
    print("Use: ravnest pull <model-id> to pre-download a model.")


def cmd_pull(args):
    """Pre-download a model from HuggingFace."""
    model_id = args.model
    cache_dir = os.environ.get("MODEL_CACHE", os.path.expanduser("~/.cache/ravnest/models"))
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading {model_id} to {cache_dir}...")
    print("(This may take a while for large models)")
    print()

    try:
        subprocess.run(
            [sys.executable, "-c", f"""
import os
os.environ['HF_HOME'] = '{cache_dir}'
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('{model_id}', cache_dir='{cache_dir}')
print('Downloading model...')
AutoModelForCausalLM.from_pretrained('{model_id}', cache_dir='{cache_dir}')
print('Done!')
"""],
            check=True,
        )
    except subprocess.CalledProcessError:
        print("Download failed. Check your network connection and HuggingFace access.")
        sys.exit(1)
    except FileNotFoundError:
        print("Python not found. Install transformers: pip install transformers")
        sys.exit(1)

    print()
    print(f"Model cached at: {cache_dir}")
    print(f"Use: ravnest up --model {model_id}")


def cmd_bench(args):
    """Benchmark inference performance against a running API."""
    import json as json_mod

    url = args.url
    tokens = args.tokens
    runs = args.runs
    model = args.model or "ravnest"

    try:
        import requests as req
    except ImportError:
        print("Install requests: pip install requests")
        sys.exit(1)

    # Check API is reachable
    try:
        resp = req.get(f"{url}/health", timeout=5)
        if resp.status_code != 200:
            print(f"API at {url} returned {resp.status_code}")
            sys.exit(1)
    except req.ConnectionError:
        print(f"Cannot reach API at {url}. Is ravnest running?")
        sys.exit(1)

    print(f"Ravnest Benchmark")
    print(f"  API:        {url}")
    print(f"  Max tokens: {tokens}")
    print(f"  Runs:       {runs}")
    print()

    import time as time_mod

    prompts = [
        "Explain what distributed computing means in one sentence.",
        "Write a haiku about artificial intelligence.",
        "What is the capital of France?",
    ]

    results = []
    for i in range(runs):
        prompt = prompts[i % len(prompts)]
        print(f"Run {i+1}/{runs}: \"{prompt[:50]}...\"")

        start = time_mod.time()
        try:
            resp = req.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": tokens,
                },
                timeout=600,
            )
            elapsed = time_mod.time() - start
            data = resp.json()

            if resp.status_code != 200:
                print(f"  Error: HTTP {resp.status_code}")
                continue

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            content = data["choices"][0]["message"]["content"]

            tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
            time_per_token = (elapsed / completion_tokens * 1000) if completion_tokens > 0 else 0

            results.append({
                "elapsed": elapsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_per_sec": tokens_per_sec,
                "time_per_token_ms": time_per_token,
            })

            print(f"  {completion_tokens} tokens in {elapsed:.2f}s "
                  f"({tokens_per_sec:.2f} tok/s, {time_per_token:.0f}ms/tok)")
            print(f"  Output: \"{content[:60]}...\"")

        except Exception as e:
            print(f"  Error: {e}")

        print()

    if not results:
        print("No successful runs.")
        sys.exit(1)

    # Summary
    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_tpt = sum(r["time_per_token_ms"] for r in results) / len(results)
    avg_elapsed = sum(r["elapsed"] for r in results) / len(results)
    total_tokens = sum(r["completion_tokens"] for r in results)

    print("=" * 50)
    print(f"BENCHMARK RESULTS ({len(results)} runs)")
    print(f"  Avg tokens/sec:     {avg_tps:.2f}")
    print(f"  Avg ms/token:       {avg_tpt:.0f}")
    print(f"  Avg response time:  {avg_elapsed:.2f}s")
    print(f"  Total tokens:       {total_tokens}")
    print("=" * 50)

    # Save results
    results_dir = os.path.expanduser("~/.cache/ravnest/benchmarks")
    os.makedirs(results_dir, exist_ok=True)
    import time as time_mod2
    results_file = os.path.join(results_dir, f"bench-{int(time_mod2.time())}.json")
    with open(results_file, "w") as f:
        json_mod.dump({
            "url": url,
            "max_tokens": tokens,
            "runs": len(results),
            "avg_tokens_per_sec": round(avg_tps, 2),
            "avg_ms_per_token": round(avg_tpt, 0),
            "avg_response_time": round(avg_elapsed, 2),
            "results": results,
        }, f, indent=2)
    print(f"Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        prog="ravnest",
        description="Ravnest distributed inference CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ravnest up
    up_parser = subparsers.add_parser("up", help="Start distributed inference")
    up_parser.add_argument("--model", "-m", type=str, default=None,
                          help="HuggingFace model ID (default: auto-select based on device)")
    up_parser.add_argument("--nodes", "-n", type=int, default=2,
                          help="Number of pipeline stages on this machine (default: 2)")
    up_parser.add_argument("--device", "-d", type=str, default=None,
                          choices=["cpu", "cuda", "mps"],
                          help="Device type (default: auto-detect)")
    up_parser.add_argument("--port", "-p", type=int, default=8000,
                          help="API port (default: 8000)")
    up_parser.add_argument("--api-key", "-k", type=str, default=None,
                          help="API key for Bearer auth (default: no auth)")
    up_parser.add_argument("--master-addr", type=str, default=None,
                          help="IP address for cross-machine mode (enables host networking)")
    up_parser.add_argument("--world-size", "-w", type=int, default=None,
                          help="Total nodes across all machines (cross-machine only, default: same as --nodes)")
    up_parser.add_argument("--proportions", type=str, default=None,
                          help="Layer split proportions per node, e.g. '0.3,0.7' (default: equal split)")
    up_parser.add_argument("--auto-profile", action="store_true", default=False,
                          help="Auto-detect hardware and compute proportions (default for cross-machine)")

    # ravnest join
    join_parser = subparsers.add_parser("join", help="Join an existing cluster from another machine")
    join_parser.add_argument("--master-addr", type=str, required=True,
                            help="IP address of the root node")
    join_parser.add_argument("--model", "-m", type=str, default=None,
                            help="HuggingFace model ID (must match the root)")
    join_parser.add_argument("--rank", "-r", type=int, default=1,
                            help="Rank of this node (default: 1)")
    join_parser.add_argument("--world-size", "-w", type=int, default=2,
                            help="Total number of nodes in the cluster (default: 2)")
    join_parser.add_argument("--device", "-d", type=str, default=None,
                            choices=["cpu", "cuda"],
                            help="Device type (default: auto-detect)")
    join_parser.add_argument("--proportions", type=str, default=None,
                            help="Layer split proportions per node, e.g. '0.3,0.7' (must match root)")
    join_parser.add_argument("--auto-profile", action="store_true", default=False,
                            help="Auto-detect hardware and compute proportions (must match root)")

    # ravnest coordinator
    coord_parser = subparsers.add_parser("coordinator",
                                         help="Run cluster coordinator (dynamic node management)")
    coord_parser.add_argument("--port", "-p", type=int, default=8080,
                             help="Coordinator port (default: 8080)")
    coord_parser.add_argument("--min-nodes", type=int, default=2,
                             help="Minimum nodes before cluster is ready (default: 2)")
    coord_parser.add_argument("--heartbeat-timeout", type=int, default=60,
                             help="Seconds before a node is considered dead (default: 60)")
    coord_parser.add_argument("--model", "-m", type=str, default=None,
                             help="HuggingFace model ID for the cluster")
    coord_parser.add_argument("--device", "-d", type=str, default=None,
                             choices=["cpu", "cuda"],
                             help="Device type for inference nodes")

    # ravnest worker
    worker_parser = subparsers.add_parser("worker",
                                          help="Join cluster via coordinator (auto-managed)")
    worker_parser.add_argument("--coordinator", "-c", type=str, required=True,
                              help="Coordinator URL (e.g. http://192.168.1.100:8080)")
    worker_parser.add_argument("--model", "-m", type=str, default=None,
                              help="HuggingFace model ID (overrides coordinator)")
    worker_parser.add_argument("--device", "-d", type=str, default=None,
                              choices=["cpu", "cuda"],
                              help="Device type (default: auto-detect)")

    # ravnest models
    subparsers.add_parser("models", help="List supported models with sizes and requirements")

    # ravnest pull
    pull_parser = subparsers.add_parser("pull", help="Pre-download a model for offline use")
    pull_parser.add_argument("model", type=str, help="HuggingFace model ID")

    # ravnest bench
    bench_parser = subparsers.add_parser("bench", help="Benchmark inference performance")
    bench_parser.add_argument("--model", "-m", type=str, default=None,
                             help="HuggingFace model ID (default: TinyLlama for CPU)")
    bench_parser.add_argument("--tokens", "-t", type=int, default=20,
                             help="Number of tokens to generate (default: 20)")
    bench_parser.add_argument("--runs", "-r", type=int, default=3,
                             help="Number of runs to average (default: 3)")
    bench_parser.add_argument("--url", type=str, default="http://localhost:8000",
                             help="API URL to benchmark (default: http://localhost:8000)")

    # ravnest native (no Docker — for Apple Silicon / MPS)
    native_parser = subparsers.add_parser(
        "native",
        help="Run a 2-node cluster natively (no Docker). Needed for macOS/MPS GPU.",
    )
    native_parser.add_argument("--model", "-m", type=str, default=None,
                               help="HuggingFace model ID (default: TinyLlama)")
    native_parser.add_argument("--nodes", "-n", type=int, default=2,
                               help="[single-machine] pipeline stages on this host (default: 2)")
    native_parser.add_argument("--device", "-d", type=str, default=None,
                               choices=["cpu", "cuda", "mps"],
                               help="Device type (default: auto-detect, prefers mps on Mac)")
    native_parser.add_argument("--port", "-p", type=int, default=8000,
                               help="API port for rank 0 (default: 8000)")
    native_parser.add_argument("--api-key", "-k", type=str, default=None,
                               help="Bearer token for API auth (default: none)")
    native_parser.add_argument("--peers", type=str, default=None,
                               help="[multi-machine] comma-separated host list, "
                                    "e.g. '192.168.1.5,192.168.1.10'. "
                                    "Must match on all machines.")
    native_parser.add_argument("--rank", type=int, default=None,
                               help="[multi-machine] this machine's rank in --peers "
                                    "(0 = root+API, N-1 = leaf)")
    native_parser.add_argument("--world-size", type=int, default=None,
                               help="[multi-machine] total nodes (default: len(peers))")
    native_parser.add_argument("--auto-profile", action="store_true", default=False,
                               help="[multi-machine] auto-detect hardware and split layers "
                                    "proportionally (MPS/GPU nodes get more layers)")

    # ravnest down / status
    subparsers.add_parser("down", help="Stop distributed inference")
    subparsers.add_parser("status", help="Show running containers")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "up":
        cmd_up(args)
    elif args.command == "join":
        cmd_join(args)
    elif args.command == "coordinator":
        cmd_coordinator(args)
    elif args.command == "worker":
        cmd_worker(args)
    elif args.command == "native":
        cmd_native(args)
    elif args.command == "models":
        cmd_models(args)
    elif args.command == "pull":
        cmd_pull(args)
    elif args.command == "bench":
        cmd_bench(args)
    elif args.command == "down":
        cmd_down(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
