<div align="center">
      <h1> Ravnest </h1>
</div>

[![Documentation Status](https://readthedocs.org/projects/ravnest/badge/?version=latest&style=for-the-badge)](http://ravnest.readthedocs.io)
[![Documentation Status](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2401.01728)

Ravnest introduces a novel asynchronous parallel training approach that combines the best aspects of data and model parallelism. This method enables the distributed training of complex deep learning models across large datasets, utilizing clusters of heterogeneous consumer-grade PCs connected via the internet. Designed with scalability and performance as key objectives, Ravnest seeks to empower researchers and machine learning practitioners. It simplifies the development and deployment of deep learning models, paving the way for innovative research and practical real-world applications.

**Documentation**: https://ravnest.readthedocs.io

**Research Paper**: https://arxiv.org/abs/2401.01728


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

### Features

- **Distributed LLM Inference** — split one model across multiple machines, each runs a slice
- **Supported Models**: Llama, Mistral, Phi-3, Qwen-2, TinyLlama
- **OpenAI-compatible API** — works with Open WebUI, LangChain, Continue.dev, any OpenAI client
- **Apple Silicon (MPS) support** — native mode uses Mac GPU, no Docker needed
- **Multi-machine over Tailscale/LAN** — combine GPUs across friends' computers
- **Auto hardware profiling** — detects GPU/CPU/MPS, splits layers proportionally
- **Docker Compose** for Linux/NVIDIA setups (CPU and GPU images)
- **Dynamic TCP backend** — hot-swap nodes without restarting the cluster
- **KV Cache** with paged attention for memory-efficient generation
- **Distributed Training** across heterogeneous consumer-grade PCs

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

### Quick Start

#### Option 1: macOS / Apple Silicon (native, no Docker)

**Requires Python 3.11+.** macOS may ship with 3.9 — install a newer version
from [python.org](https://www.python.org/downloads/macos/) or `brew install python@3.12`.

```bash
git clone https://github.com/ravenprotocol/ravnest-agent.git
cd ravnest-agent && git checkout llm_optim

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[inference]'

ravnest native                     # auto-detects MPS, starts 2 local nodes
```

Everything installs into `.venv/`. To remove: `rm -rf ravnest-agent`.

> If `pip install` fails with **"externally-managed-environment"**, the
> `venv` step above is the fix. Do **not** use `--break-system-packages`.

#### Option 2: Linux with Docker (GPU or CPU)

```bash
ravnest up                                                    # auto-detect GPU/CPU
ravnest up --model meta-llama/Llama-3.1-8B --nodes 3         # custom model
ravnest down                                                  # stop
```

Or use Docker Compose directly:
```bash
cd deploy
docker compose up --build                                     # GPU (NVIDIA)
docker compose -f docker-compose.cpu.yml up --build           # CPU
```

#### Option 3: One-command install (Linux)

```bash
curl -fsSL https://raw.githubusercontent.com/ravenprotocol/ravnest-agent/llm_optim/install.sh | bash
```

Auto-detects GPU/CPU, pulls Docker images, starts a 2-node cluster.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

### Multi-Machine Inference

Split a model across two (or more) computers on the same network or
connected via [Tailscale](https://tailscale.com/download).

**Machine A** (root — runs the API, gets more layers if it has a GPU):
```bash
ravnest native --peers 192.168.1.5,192.168.1.10 --rank 0
```

**Machine B** (leaf — receives activations, runs its slice):
```bash
ravnest native --peers 192.168.1.5,192.168.1.10 --rank 1
```

Replace IPs with your actual LAN IPs (`ipconfig getifaddr en0` on Mac,
`hostname -I` on Linux) or Tailscale IPs (`tailscale ip -4`).

Add `--auto-profile` to both commands to automatically give faster nodes
(GPU/MPS) more layers than CPU nodes.

Start the **leaf first**, then the **root**. The root waits up to 60 seconds
for the leaf to come online and prints progress while waiting.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

### Using the API

Once running, send requests to the OpenAI-compatible endpoint:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ravnest",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
  }'
```

Compatible with **Open WebUI**, **LangChain**, **Continue.dev**, and any
OpenAI client — just point it at `http://localhost:8000`.

#### Streaming

Add `"stream": true` to your request for token-by-token Server-Sent Events:

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ravnest",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "max_tokens": 200,
    "stream": true
  }'
```

#### Open WebUI (ChatGPT-like interface)

Point [Open WebUI](https://github.com/open-webui/open-webui) at your
Ravnest API for a full chat UI:

```bash
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=unused \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

Then open http://localhost:3000. On first visit, create a local account
(stays on your machine). Select "ravnest" as the model and start chatting.

> On Linux, replace `host.docker.internal` with your machine's LAN IP.

#### Benchmarking

```bash
ravnest bench --tokens 50 --runs 3
```

#### Background mode

```bash
ravnest native --background        # starts cluster, returns to shell
ravnest native-stop                # stops background cluster
```

#### Other CLI commands

```bash
ravnest models                     # list supported models with sizes
ravnest pull <model-id>            # pre-download a model
ravnest bench --url http://...     # benchmark tok/s against a running API
ravnest status                     # show running Docker containers
ravnest down                       # stop Docker containers
```

See [deploy/README.md](deploy/README.md) for full Docker details.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

### Distributed Training

Clone the Repository:
```bash
git clone https://github.com/ravenprotocol/ravnest.git
```

Generate the submodel files:

```bash
python cluster_formation.py
```

> **_NOTE:_**  Uncomment the correct lines in ```cluster_formation.py``` for CNN/ResNet-50/Inception-V3/GPT-Sorter/BERT models.

Execution of Clients (in 3 terminals) for CNN:

Create 3 copies of the ``provider.py`` file inside ``examples/cnn/`` folder. Rename these files as ``provider_0.py``, ``provider_1.py`` and ``provider_2.py``. In each of these files, set the ``name`` parameter of ``Node()`` object to ``'node_0'``, ``'node_1'`` and ``'node_2'``.

```bash
python examples/cnn/provider_0.py
```
```bash
python examples/cnn/provider_1.py
```
```bash
python examples/cnn/provider_2.py
```

> **_NOTE:_** If you have installed Ravnest via Pip, you will have to delete the entire ``ravnest`` subfolder in your cloned directory so that your scripts utilize methods and classes pointing to the pip installed library.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

### Citation
If you have found Ravnest or its foundational components and algorithms to be beneficial in your research, please consider citing the following source:

```
@misc{menon2024ravnest,
      title={Ravnest: Decentralized Asynchronous Training on Heterogeneous Devices}, 
      author={Anirudh Rajiv Menon and Unnikrishnan Menon and Kailash Ahirwar},
      year={2024},
      eprint={2401.01728},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
