#!/bin/bash
# Ravnest — Distributed LLM inference in one command
# Usage: curl -fsSL https://raw.githubusercontent.com/ravenprotocol/ravnest-agent/llm_optim/install.sh | bash
set -e

REPO="ravnest/ravnest"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}  ╔══════════════════════════════════════╗${NC}"
echo -e "${BOLD}  ║     Ravnest Distributed Inference     ║${NC}"
echo -e "${BOLD}  ║   Run LLMs across multiple machines   ║${NC}"
echo -e "${BOLD}  ╚══════════════════════════════════════╝${NC}"
echo ""

# --- Check Docker ---
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found.${NC}"
    echo ""
    echo "Install Docker first:"
    echo -e "  ${BOLD}curl -fsSL https://get.docker.com | sh${NC}"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

if ! docker info &> /dev/null 2>&1; then
    echo -e "${RED}Docker is installed but not running (or needs sudo).${NC}"
    echo ""
    echo "Try: sudo systemctl start docker"
    echo "Or:  sudo usermod -aG docker \$USER && newgrp docker"
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker found"

# --- Detect GPU ---
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_GPU=true
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo -e "${GREEN}✓${NC} GPU detected: $GPU_NAME"
    fi
fi

if [ "$HAS_GPU" = false ]; then
    echo -e "${YELLOW}!${NC} No NVIDIA GPU detected — running in CPU mode (slower but works)"
fi

# --- Pick image and model ---
if [ "$HAS_GPU" = true ]; then
    IMAGE="$REPO:gpu-latest"
    MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
    DEVICE="cuda"
    COMPOSE_EXTRA='    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]'
else
    IMAGE="$REPO:cpu-latest"
    MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
    DEVICE="cpu"
    COMPOSE_EXTRA=""
fi

PORT="${PORT:-8000}"

echo -e "${GREEN}✓${NC} Model: $MODEL"
echo -e "${GREEN}✓${NC} Device: $DEVICE"
echo -e "${GREEN}✓${NC} API port: $PORT"
echo ""

# --- Pull image (or build from source) ---
echo "Pulling image ($IMAGE)..."
if ! docker pull "$IMAGE" 2>/dev/null; then
    echo -e "${YELLOW}!${NC} Pre-built image not available yet. Building from source..."
    echo ""

    BUILD_DIR=$(mktemp -d)
    git clone --depth 1 https://github.com/ravenprotocol/ravnest-agent.git "$BUILD_DIR/ravnest" 2>&1 | tail -2

    if [ "$HAS_GPU" = true ]; then
        docker build -f "$BUILD_DIR/ravnest/deploy/Dockerfile" -t "$IMAGE" "$BUILD_DIR/ravnest" 2>&1 | tail -5
    else
        docker build -f "$BUILD_DIR/ravnest/deploy/Dockerfile.cpu" -t "$IMAGE" "$BUILD_DIR/ravnest" 2>&1 | tail -5
    fi

    rm -rf "$BUILD_DIR"
    echo -e "${GREEN}✓${NC} Built from source"
fi
echo ""

# --- Create compose file ---
RAVNEST_DIR="$HOME/.ravnest"
mkdir -p "$RAVNEST_DIR"

cat > "$RAVNEST_DIR/docker-compose.yml" << COMPOSE_EOF
services:
  node-0:
    image: $IMAGE
    environment:
      - RANK=0
      - WORLD_SIZE=2
      - MASTER_ADDR=node-0
      - MASTER_PORT=29500
      - MODEL_NAME=$MODEL
      - NODE_ROLE=root
      - RAVNEST_DEVICE=$DEVICE
      - RAVNEST_BACKEND=dynamic
      - RAVNEST_PEERS=node-0,node-1
      - PYTHONUNBUFFERED=1
    volumes:
      - model_cache:/app/model_cache
    ports:
      - "$PORT:8000"
    command: python deploy/entrypoint.py
$COMPOSE_EXTRA

  node-1:
    image: $IMAGE
    environment:
      - RANK=1
      - WORLD_SIZE=2
      - MASTER_ADDR=node-0
      - MASTER_PORT=29500
      - MODEL_NAME=$MODEL
      - NODE_ROLE=leaf
      - RAVNEST_DEVICE=$DEVICE
      - RAVNEST_BACKEND=dynamic
      - RAVNEST_PEERS=node-0,node-1
      - PYTHONUNBUFFERED=1
    volumes:
      - model_cache:/app/model_cache
    command: python deploy/entrypoint.py
    depends_on:
      - node-0
$COMPOSE_EXTRA

volumes:
  model_cache:
COMPOSE_EOF

echo -e "${GREEN}✓${NC} Compose file created at $RAVNEST_DIR/docker-compose.yml"
echo ""

# --- Start ---
echo -e "${BOLD}Starting Ravnest...${NC}"
echo "(First run downloads the model — this takes a few minutes)"
echo ""

cd "$RAVNEST_DIR"
docker compose up -d 2>&1 | tail -5

echo ""
echo -e "${BOLD}Waiting for API to be ready...${NC}"
for i in $(seq 1 60); do
    if curl -s http://localhost:$PORT/health 2>/dev/null | grep -q "ok"; then
        echo ""
        echo -e "${GREEN}══════════════════════════════════════${NC}"
        echo -e "${GREEN}  Ravnest is running!${NC}"
        echo -e "${GREEN}══════════════════════════════════════${NC}"
        echo ""
        echo "  API:  http://localhost:$PORT"
        echo ""
        echo -e "  ${BOLD}Try it:${NC}"
        echo ""
        echo "  curl -X POST http://localhost:$PORT/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\":\"ravnest\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":50}'"
        echo ""
        echo -e "  ${BOLD}Commands:${NC}"
        echo "  docker compose -f $RAVNEST_DIR/docker-compose.yml logs -f    # view logs"
        echo "  docker compose -f $RAVNEST_DIR/docker-compose.yml down       # stop"
        echo ""
        echo "  Works with Open WebUI, LangChain, Continue.dev —"
        echo "  point any OpenAI-compatible tool at http://localhost:$PORT"
        echo ""
        exit 0
    fi
    sleep 5
    echo -n "."
done

echo ""
echo -e "${YELLOW}API not ready yet. The model is still downloading.${NC}"
echo ""
echo "Check progress:"
echo "  docker compose -f $RAVNEST_DIR/docker-compose.yml logs -f"
echo ""
echo "Once you see 'Uvicorn running', test with:"
echo "  curl http://localhost:$PORT/health"
