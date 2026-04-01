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
                          choices=["cpu", "cuda"],
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
    elif args.command == "down":
        cmd_down(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
