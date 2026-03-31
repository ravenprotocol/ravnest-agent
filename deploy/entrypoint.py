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

# Add project root to path
sys.path.insert(0, "/app")

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
    device_str = os.environ.get("RAVNEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    use_cpu = device_str == "cpu"

    print(f"[node-{rank}] Starting as {role}, model={model_name}, device={device_str}")

    # Download tokenizer (both nodes need it for decode)
    print(f"[node-{rank}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Load model with lazy init (both nodes need full checkpoint to determine their layers)
    print(f"[node-{rank}] Loading model with lazy init...")
    dtype = torch.float32 if use_cpu else torch.float16
    init_ctx = LazyInitContext()
    with init_ctx:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cpu" if use_cpu else "cuda",
            cache_dir=cache_dir,
        )
    model.eval()

    # Use longer timeout for CPU (inference is slow)
    timeout_minutes = 30 if use_cpu else 10

    # Parse proportions from env (e.g. "0.3,0.7")
    proportions = None
    prop_str = os.environ.get("RAVNEST_PROPORTIONS", "")
    if prop_str:
        proportions = [float(p) for p in prop_str.split(",")]
        print(f"[node-{rank}] Using proportional split: {proportions}")

    print(f"[node-{rank}] Creating Node (reduce_factor=1, backend=gloo, timeout={timeout_minutes}min)...")
    node = Node(
        model=model,
        device=device,
        dtype="float32" if use_cpu else "float16",
        batch_size=1,
        mode="inference",
        seq_length=5,
        backend="gloo",
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
    print("[node-0] Starting API server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


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
