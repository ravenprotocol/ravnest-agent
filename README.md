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

- **Distributed Training** across heterogeneous consumer-grade PCs
- **Distributed LLM Inference** with paged attention and pipeline parallelism
- **Supported Models**: Llama, Mistral, Phi-3, Qwen-2
- **Docker Compose** setup for multi-node inference with OpenAI-compatible API
- **Dual backends**: gRPC for TCP, torch.distributed (Gloo/NCCL) for GPU clusters
- **KV Cache** with paged attention for memory-efficient inference

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

### Installation
```bash
pip install git+https://github.com/ravenprotocol/ravnest.git
```

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

### Distributed Inference

Run a Llama model split across multiple containers with an OpenAI-compatible API endpoint.

**One command** (auto-detects GPU/CPU):
```bash
ravnest up
```

**With options:**
```bash
ravnest up --model meta-llama/Llama-3.1-8B --nodes 3 --port 8000
ravnest up --device cpu --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
ravnest status
ravnest down
```

**Or use Docker Compose directly:**

```bash
cd deploy
docker compose up --build                                    # GPU
docker compose -f docker-compose.cpu.yml up --build          # CPU
```

Once running, send requests to the API:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ravnest",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
  }'
```

The API is compatible with Open WebUI, LangChain, Continue.dev, and any tool that speaks the OpenAI protocol. See [deploy/README.md](deploy/README.md) for full details.

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
