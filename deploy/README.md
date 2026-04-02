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
- **Llama** — Llama-3.2 (1B, 3B), Llama-3.1 (8B, 8B-Instruct), TinyLlama
- **Mistral** — Mistral-7B, Ministral-3B
- **Phi** — Phi-3-mini (3.8B), Phi-3.5, Phi-2
- **Qwen-2** — Qwen2-1.5B, Qwen2-7B

## Dynamic Cluster (Coordinator Mode)

For communities or teams, the coordinator manages nodes dynamically. Workers register,
hardware is auto-profiled, and layers are distributed proportionally. When nodes
join or leave, the cluster reconfigures automatically via barrier synchronization.

**Machine 1 (coordinator):**
```bash
ravnest coordinator --min-nodes 2 --model meta-llama/Llama-3.2-3B
```

**Machine 2+ (workers):**
```bash
ravnest worker --coordinator http://machine1:8080
```

The coordinator:
- Auto-detects each worker's hardware (GPU VRAM or CPU RAM)
- Computes optimal layer proportions (GPU memory weighted 10x vs CPU)
- Distributes configuration to all workers
- Barrier-synchronizes reconfiguration when nodes join or leave
- Workers hot-swap: model stays in memory, only layers and connections rebuild

```
  Coordinator (:8080)
  ├── /register — workers join
  ├── /config   — workers get rank, proportions, peer IPs
  ├── /ready    — workers signal ready for reconfigure
  ├── /barrier  — workers poll until all ready
  └── /status   — cluster overview

  Worker 0 (RTX 3090)  →  layers 0-18   (71%)
  Worker 1 (CPU 16GB)  →  layers 18-19  (5%)
  Worker 2 (RTX 4060)  →  layers 19-22  (24%)
```

## Running Tests

```bash
# Basic smoke test (2-node Docker Compose)
bash deploy/test.sh

# With auth
RAVNEST_API_KEY=my-secret bash deploy/test.sh

# Coordinator test suite (registration, barrier, proportions, integration)
bash deploy/test_coordinator.sh
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
Machines must be able to reach each other on port 29500 (Gloo), port 29400 (auto-profile),
and port 8000 (API, root only).

## Networking Guide

Machines need to reach each other directly. Here's how depending on your setup:

### Same LAN (home/office)

Already works. Use your local IP (`ifconfig` or `ip addr`).

### Different networks — Tailscale (easiest, free for up to 3 users)

[Tailscale](https://tailscale.com) is a mesh VPN. One command to install, zero config.
Every machine gets a stable IP (100.x.y.z) and can reach every other machine directly.

```bash
# On every machine (Linux)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Check your Tailscale IP
tailscale ip -4
# Example: 100.64.0.1

# Machine 1
ravnest up --master-addr 100.64.0.1 --world-size 2

# Machine 2
ravnest join --master-addr 100.64.0.1 --world-size 2
```

Good for: first tests, small teams (up to 3 users free).

### Different networks — Nebula (free, unlimited users, self-hosted)

[Nebula](https://github.com/slackhq/nebula) is an open-source mesh VPN built by Slack.
You run a "lighthouse" server that helps machines find each other, then traffic flows
directly between machines. No user limits, no cost.

**1. Set up a lighthouse** (one-time, any cheap VPS):

```bash
# Download Nebula
curl -fsSL https://github.com/slackhq/nebula/releases/latest/download/nebula-linux-amd64.tar.gz | tar xz

# Create a certificate authority
./nebula-cert ca -name "ravnest-cluster"

# Create lighthouse certificate
./nebula-cert sign -name lighthouse -ip 10.42.0.1/24

# Create config (lighthouse.yml):
# pki:
#   ca: /etc/nebula/ca.crt
#   cert: /etc/nebula/lighthouse.crt
#   key: /etc/nebula/lighthouse.key
# lighthouse:
#   am_lighthouse: true
# listen:
#   host: 0.0.0.0
#   port: 4242
# firewall:
#   inbound:
#     - port: any
#       proto: any
#       host: any
#   outbound:
#     - port: any
#       proto: any
#       host: any

./nebula -config lighthouse.yml
```

**2. For each community member:**

```bash
# On the lighthouse, generate a cert for each member:
./nebula-cert sign -name "alice" -ip 10.42.0.2/24
./nebula-cert sign -name "bob" -ip 10.42.0.3/24
# Send them: ca.crt + their .crt + .key files

# On each member's machine:
curl -fsSL https://github.com/slackhq/nebula/releases/latest/download/nebula-linux-amd64.tar.gz | tar xz

# config.yml:
# pki:
#   ca: /etc/nebula/ca.crt
#   cert: /etc/nebula/alice.crt
#   key: /etc/nebula/alice.key
# static_host_map:
#   "10.42.0.1": ["<lighthouse-public-ip>:4242"]
# lighthouse:
#   hosts:
#     - 10.42.0.1
# listen:
#   host: 0.0.0.0
#   port: 4242
# firewall:
#   inbound:
#     - port: any
#       proto: any
#       host: any
#   outbound:
#     - port: any
#       proto: any
#       host: any

sudo ./nebula -config config.yml
```

**3. Run Ravnest:**

```bash
# Alice (root, 10.42.0.2)
ravnest up --master-addr 10.42.0.2 --world-size 3

# Bob (10.42.0.3)
ravnest join --master-addr 10.42.0.2 --world-size 3 --rank 1

# Carol (10.42.0.4)
ravnest join --master-addr 10.42.0.2 --world-size 3 --rank 2
```

Good for: communities of 10-100+ people, no cost, full control.

### Which to use?

| | Tailscale | Nebula |
|---|-----------|--------|
| Setup | 1 command | 30 min (lighthouse + certs) |
| Cost | Free up to 3 users | Free, unlimited |
| Server needed | No | Yes (tiny VPS) |
| Best for | First tests, small teams | Communities, long-term |

## Current Limitations

- Single request at a time (returns 503 if busy)
