"""
ravnest CLI — one-command distributed inference.

Usage:
    ravnest up [--model MODEL] [--nodes N] [--device cpu|cuda] [--port PORT]
    ravnest down
    ravnest status
"""

import argparse
import os
import shutil
import subprocess
import sys
import textwrap


def detect_hardware():
    """Detect available GPUs and return device info."""
    # Check nvidia-smi first (works without torch)
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

    # Fallback to torch if available
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


def generate_compose(model, nodes, device, port):
    """Generate a docker-compose.yml string."""
    deploy_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deploy")

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
    for rank in range(nodes):
        role = "root" if rank == 0 else "leaf"
        name = f"node-{rank}"

        env_lines = [
            f"      - RANK={rank}",
            f"      - WORLD_SIZE={nodes}",
            f"      - MASTER_ADDR=node-0",
            f"      - MASTER_PORT=29500",
            f"      - MODEL_NAME={model}",
            f"      - NODE_ROLE={role}",
            f"      - RAVNEST_DEVICE={device}",
            f"      - PYTHONUNBUFFERED=1",
        ]

        svc = f"""  {name}:
    build:
      context: ..
      dockerfile: {dockerfile}
    environment:
{chr(10).join(env_lines)}
    volumes:
      - model_cache:/app/model_cache
    command: python deploy/entrypoint.py"""

        if rank == 0:
            svc += f"""
    ports:
      - "{port}:8000\""""

        if rank > 0:
            svc += """
    depends_on:
      - node-0"""

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
    # Check if we're in the repo
    repo_deploy = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deploy")
    if os.path.isdir(repo_deploy):
        return repo_deploy
    # Check current directory
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

    print(f"Ravnest Distributed Inference")
    print(f"  Device:  {device}")
    if device == "cuda" and hw["gpus"]:
        for i, gpu in enumerate(hw["gpus"]):
            print(f"  GPU {i}:   {gpu['name']} ({gpu['vram_gb']} GB)")
    print(f"  Model:   {model}")
    print(f"  Nodes:   {nodes}")
    print(f"  API:     http://localhost:{port}")
    print()

    deploy_dir = find_deploy_dir()
    if deploy_dir is None:
        print("Error: cannot find deploy/ directory. Run from the ravnest repo root.")
        sys.exit(1)

    compose_content = generate_compose(model, nodes, device, port)
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


def main():
    parser = argparse.ArgumentParser(
        prog="ravnest",
        description="Ravnest distributed inference CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    up_parser = subparsers.add_parser("up", help="Start distributed inference")
    up_parser.add_argument("--model", "-m", type=str, default=None,
                          help="HuggingFace model ID (default: auto-select based on device)")
    up_parser.add_argument("--nodes", "-n", type=int, default=2,
                          help="Number of pipeline stages (default: 2)")
    up_parser.add_argument("--device", "-d", type=str, default=None,
                          choices=["cpu", "cuda"],
                          help="Device type (default: auto-detect)")
    up_parser.add_argument("--port", "-p", type=int, default=8000,
                          help="API port (default: 8000)")

    subparsers.add_parser("down", help="Stop distributed inference")
    subparsers.add_parser("status", help="Show running containers")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "up":
        cmd_up(args)
    elif args.command == "down":
        cmd_down(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
