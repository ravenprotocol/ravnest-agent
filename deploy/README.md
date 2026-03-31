# Ravnest Distributed Inference

Run LLMs split across multiple machines with an OpenAI-compatible API.

## Quick Start

**Using the CLI** (recommended):
```bash
pip install -e .
ravnest up                    # auto-detects GPU/CPU, picks model, starts serving
ravnest up -m meta-llama/Llama-3.1-8B -n 3 -k my-secret-key
ravnest status
ravnest down
```

**Using Docker Compose directly:**
```bash
cd deploy

# GPU (requires NVIDIA Container Toolkit)
docker compose up --build

# CPU (for testing without GPU)
docker compose -f docker-compose.cpu.yml up --build
```

## Prerequisites

**GPU setup:**
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- At least 1 NVIDIA GPU with 8GB+ VRAM
- HuggingFace account with access to gated models (Llama)

**CPU setup:**
- Docker
- ~4GB RAM per node

## API

OpenAI-compatible chat completions at `http://localhost:8000`.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming and non-streaming) |
| `/health` | GET | Health check |

### Non-streaming request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ravnest",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

### Streaming request

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ravnest",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50,
    "stream": true
  }'
```

Streaming returns Server-Sent Events in the OpenAI format:
```
data: {"choices":[{"delta":{"content":"Hello"}}]}
data: {"choices":[{"delta":{"content":" there"}}]}
data: [DONE]
```

### Authentication

Set `RAVNEST_API_KEY` to require Bearer token auth. If not set, all requests are allowed.

```bash
# With Docker Compose
RAVNEST_API_KEY=my-secret docker compose up

# With CLI
ravnest up --api-key my-secret

# Sending authenticated requests
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret" \
  -d '{"model":"ravnest","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'
```

The health endpoint (`/health`) does not require auth.

### Compatible tools

Works with any tool that speaks the OpenAI protocol. Point it at `http://localhost:8000`:
- Open WebUI
- LangChain
- Continue.dev
- LiteLLM
- Any OpenAI SDK client

## Configuration

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model, -m` | auto (Llama-3.2-3B for GPU, TinyLlama for CPU) | HuggingFace model ID |
| `--nodes, -n` | 2 | Number of pipeline stages on this machine |
| `--device, -d` | auto-detect | `cpu` or `cuda` |
| `--port, -p` | 8000 | API port |
| `--api-key, -k` | none | API key for Bearer auth |
| `--master-addr` | none | IP address for cross-machine mode |
| `--world-size, -w` | same as --nodes | Total nodes across all machines |

### Docker Compose environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `meta-llama/Llama-3.2-3B` | HuggingFace model ID |
| `WORLD_SIZE` | `2` | Number of pipeline stages |
| `MASTER_PORT` | `29500` | torch.distributed rendezvous port |
| `RAVNEST_DEVICE` | auto | `cpu` or `cuda` |
| `RAVNEST_API_KEY` | none | API key for Bearer auth |

### Supported models

Any model with a Ravnest split spec:
- Llama (1B, 3B, 8B, 8B-Instruct)
- Qwen-2
- TinyLlama (good for CPU testing)

## Running the Smoke Test

```bash
# Without auth
bash deploy/test.sh

# With auth
RAVNEST_API_KEY=my-secret bash deploy/test.sh
```

## How It Works

```
                         ┌─────────────────────────────────────┐
                         │          Docker Network              │
User ──── HTTP ────────► │                                      │
                         │  node-0 (root)      node-1 (leaf)    │
                         │  ┌─────────────┐   ┌─────────────┐  │
                         │  │ FastAPI API  │   │             │  │
                         │  │ Layers 0-N   │──►│ Layers N+1-M│  │
                         │  │             │◄──│ (feedback)   │  │
                         │  └─────────────┘   └─────────────┘  │
                         │        │                             │
                         │  shared volume (model weights)       │
                         └─────────────────────────────────────┘
```

1. Both containers download the full model, each prunes to its own layers
2. Communication uses PyTorch distributed (Gloo for CPU/cross-container, NCCL for GPU)
3. Tokens are generated one at a time: root forwards through its layers, sends activations to leaf, leaf computes and broadcasts the next token back
4. KV cache with paged attention keeps memory usage efficient

## Cross-Machine Inference

Run a model split across multiple physical machines. Each machine downloads the model
independently and connects via Gloo over the network.

**Machine 1 (root, IP: 192.168.1.100):**
```bash
ravnest up --master-addr 192.168.1.100 --nodes 1 --world-size 2 --model meta-llama/Llama-3.2-3B
```

**Machine 2 (leaf):**
```bash
ravnest join --master-addr 192.168.1.100 --rank 1 --world-size 2 --model meta-llama/Llama-3.2-3B
```

**3+ machines:**
```bash
# Machine 1 (root)
ravnest up --master-addr 192.168.1.100 --nodes 1 --world-size 3

# Machine 2
ravnest join --master-addr 192.168.1.100 --rank 1 --world-size 3

# Machine 3
ravnest join --master-addr 192.168.1.100 --rank 2 --world-size 3
```

Cross-machine mode uses `network_mode: host` so containers share the host's network.
All machines must be on the same LAN and able to reach each other on port 29500 (Gloo)
and port 8000 (API, root only).

## Current Limitations

- Single request at a time (returns 503 if busy)
