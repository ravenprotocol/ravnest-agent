"""
Entrypoint for Ravnest distributed inference Docker containers.

Node-0 (root): Downloads model, initializes pipeline, starts FastAPI API server.
Node-1 (leaf): Initializes pipeline, enters receive loop waiting for root's broadcasts.
"""

import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path (Docker uses /app; native run uses the repo dir)
if os.path.isdir("/app") and os.path.isdir("/app/ravnest"):
    sys.path.insert(0, "/app")
else:
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _repo_root)

from ravnest.lazy_init.lazy_context import LazyInitContext
from ravnest import Node
from ravnest.inference import InferenceEngine


def setup_gloo_timeout():
    """Set Gloo timeout high enough for CPU inference (which is slow)."""
    import datetime
    os.environ.setdefault("GLOO_TIMEOUT_SECONDS", "1800")  # 30 minutes


def create_node_and_engine():
    setup_gloo_timeout()
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B")
    cache_dir = "/app/model_cache"
    role = os.environ.get("NODE_ROLE", "root")
    rank = int(os.environ.get("RANK", "0"))
    def _default_device():
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    device_str = os.environ.get("RAVNEST_DEVICE", _default_device())
    device = torch.device(device_str)
    use_cpu = device_str == "cpu"
    use_mps = device_str == "mps"

    print(f"[node-{rank}] Starting as {role}, model={model_name}, device={device_str}")

    # Download tokenizer (both nodes need it for decode)
    print(f"[node-{rank}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Load model with lazy init (both nodes need full checkpoint to determine their layers)
    print(f"[node-{rank}] Loading model with lazy init...")
    # MPS supports fp16 but load on CPU first, Node.__init__ moves it.
    dtype = torch.float32 if use_cpu else torch.float16
    device_map = "cpu" if (use_cpu or use_mps) else "cuda"
    init_ctx = LazyInitContext()
    with init_ctx:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            cache_dir=cache_dir,
        )
    model.eval()

    # Use longer timeout for CPU/MPS (slower than discrete GPU)
    timeout_minutes = 30 if (use_cpu or use_mps) else 10

    # Determine layer proportions
    proportions = None
    prop_str = os.environ.get("RAVNEST_PROPORTIONS", "")
    auto_profile = os.environ.get("RAVNEST_AUTO_PROFILE", "").lower() in ("1", "true", "yes")
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    master_addr = os.environ.get("MASTER_ADDR", "node-0")

    if prop_str:
        proportions = [float(p) for p in prop_str.split(",")]
        print(f"[node-{rank}] Using manual proportions: {proportions}")
    elif auto_profile and world_size > 1:
        from ravnest.hardware import collect_hardware_as_root, report_hardware_to_root
        if rank == 0:
            hardware_list, proportions = collect_hardware_as_root(world_size)
        else:
            my_info, proportions = report_hardware_to_root(rank, master_addr)
        print(f"[node-{rank}] Auto-profiled proportions: {proportions}")

    backend = os.environ.get("RAVNEST_BACKEND", "gloo")
    print(f"[node-{rank}] Creating Node (reduce_factor=1, backend={backend}, timeout={timeout_minutes}min)...")
    node = Node(
        model=model,
        device=device,
        dtype="float32" if (use_cpu or use_mps) else "float16",
        batch_size=1,
        mode="inference",
        seq_length=5,
        backend=backend,
        cluster_length=2,
        reduce_factor=1,
        dist_timeout=timeout_minutes,
        proportions=proportions,
    )
    node.model.eval()

    print(f"[node-{rank}] Creating InferenceEngine...")
    engine = InferenceEngine(node, tokenizer)

    print(f"[node-{rank}] Ready. Layers {node.layer_start_idx}-{node.layer_end_idx}")
    return engine, tokenizer


def run_root():
    """Node-0: create engine, start FastAPI server."""
    engine, tokenizer = create_node_and_engine()

    # Import and start the API server
    from deploy.api_server import create_app

    app = create_app(engine, tokenizer)

    import uvicorn
    api_port = int(os.environ.get("RAVNEST_API_PORT", "8000"))
    print(f"[node-0] Starting API server on 0.0.0.0:{api_port}")
    uvicorn.run(app, host="0.0.0.0", port=api_port, log_level="info")


def run_leaf():
    """Node-1: create engine, enter infinite receive loop."""
    engine, tokenizer = create_node_and_engine()

    print("[node-1] Entering receive loop (waiting for root broadcasts)...")
    while True:
        try:
            engine.generate(prompt_list=None, max_seq_lengths=None)
        except RuntimeError as e:
            import traceback
            print(f"[node-1] Generation error: {e}")
            traceback.print_exc()
            time.sleep(5)
        except Exception as e:
            import traceback
            print(f"[node-1] Unexpected error: {e}")
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    role = os.environ.get("NODE_ROLE", "root")
    if role == "root":
        run_root()
    else:
        run_leaf()
