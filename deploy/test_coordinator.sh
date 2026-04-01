#!/bin/bash
# Test suite for coordinator, workers, barrier, and dynamic cluster management
set -e

PASS=0
FAIL=0
COORD_URL="http://localhost:8080"
API_URL="http://localhost:8000"

check() {
    local name="$1"
    local expected="$2"
    local actual="$3"
    if [ "$actual" = "$expected" ]; then
        echo "PASS: $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $name (expected '$expected', got '$actual')"
        FAIL=$((FAIL + 1))
    fi
}

cleanup() {
    echo "Cleaning up..."
    docker stop rn-coord rn-w0 rn-w1 rn-w2 2>/dev/null
    docker rm rn-coord rn-w0 rn-w1 rn-w2 2>/dev/null
    docker network rm ravnest-test-net 2>/dev/null
}
trap cleanup EXIT

echo "=== Ravnest Coordinator Test Suite ==="
echo ""

# Build image
echo "--- Building image ---"
docker build -f deploy/Dockerfile.cpu -t ravnest-test -q .. 2>&1 | tail -1

# Setup network
docker network create ravnest-test-net 2>/dev/null || true

# --- Test 1: Coordinator starts and reports empty ---
echo ""
echo "--- Coordinator Startup ---"
docker run -d --name rn-coord --network ravnest-test-net -p 8080:8080 \
  ravnest-test python ravnest/cli.py coordinator --port 8080 --min-nodes 2 \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cpu >/dev/null
sleep 3

STATUS=$(curl -s $COORD_URL/status)
NODES=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['num_nodes'])")
READY=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['ready'])")
check "Coordinator starts with 0 nodes" "0" "$NODES"
check "Coordinator not ready" "False" "$READY"

# --- Test 2: Node registration ---
echo ""
echo "--- Node Registration ---"
REG=$(curl -s -X POST $COORD_URL/register \
  -H "Content-Type: application/json" \
  -d '{"node_id":"10.0.0.1","hardware":{"type":"cuda","name":"RTX 3090","memory_gb":24.0}}')
NODES=$(echo "$REG" | python3 -c "import sys,json; print(json.load(sys.stdin)['num_nodes'])")
check "First node registered" "1" "$NODES"

REG=$(curl -s -X POST $COORD_URL/register \
  -H "Content-Type: application/json" \
  -d '{"node_id":"10.0.0.2","hardware":{"type":"cpu","name":"CPU","memory_gb":16.0}}')
READY=$(echo "$REG" | python3 -c "import sys,json; print(json.load(sys.stdin)['ready'])")
check "Cluster ready with 2 nodes" "True" "$READY"

# --- Test 3: Proportions ---
echo ""
echo "--- Proportions ---"
PROPS=$(echo "$REG" | python3 -c "import sys,json; print(json.load(sys.stdin)['proportions'])")
check "GPU gets more layers than CPU" "[0.94, 0.06]" "$PROPS"

# --- Test 4: Config endpoint ---
echo ""
echo "--- Config ---"
CONFIG=$(curl -s $COORD_URL/config/10.0.0.1)
RANK=$(echo "$CONFIG" | python3 -c "import sys,json; print(json.load(sys.stdin)['rank'])")
PEERS=$(echo "$CONFIG" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['peers']))")
check "GPU node gets rank 0" "0" "$RANK"
check "Config includes 2 peers" "2" "$PEERS"

# --- Test 5: Unknown node returns 404 ---
echo ""
echo "--- Error Handling ---"
CODE=$(curl -s -o /dev/null -w "%{http_code}" $COORD_URL/config/unknown)
check "Unknown node returns 404" "404" "$CODE"

# --- Test 6: Barrier ---
echo ""
echo "--- Barrier ---"
VERSION=$(echo "$REG" | python3 -c "import sys,json; print(json.load(sys.stdin)['cluster_version'])")

BARRIER=$(curl -s $COORD_URL/barrier/$VERSION)
ALL_READY=$(echo "$BARRIER" | python3 -c "import sys,json; print(json.load(sys.stdin)['all_ready'])")
check "Barrier not ready (no signals)" "False" "$ALL_READY"

curl -s -X POST $COORD_URL/ready/10.0.0.1/$VERSION >/dev/null
curl -s -X POST $COORD_URL/ready/10.0.0.2/$VERSION >/dev/null
BARRIER=$(curl -s $COORD_URL/barrier/$VERSION)
ALL_READY=$(echo "$BARRIER" | python3 -c "import sys,json; print(json.load(sys.stdin)['all_ready'])")
check "Barrier cleared after all signal ready" "True" "$ALL_READY"

# --- Test 7: Node leave triggers recompute ---
echo ""
echo "--- Node Leave ---"
LEAVE=$(curl -s -X POST $COORD_URL/leave/10.0.0.2)
NODES=$(echo "$LEAVE" | python3 -c "import sys,json; print(json.load(sys.stdin)['num_nodes'])")
check "Node removed" "1" "$NODES"

# --- Test 8: 3rd node join ---
echo ""
echo "--- Dynamic Join ---"
curl -s -X POST $COORD_URL/register \
  -H "Content-Type: application/json" \
  -d '{"node_id":"10.0.0.2","hardware":{"type":"cpu","name":"CPU","memory_gb":16.0}}' >/dev/null
curl -s -X POST $COORD_URL/register \
  -H "Content-Type: application/json" \
  -d '{"node_id":"10.0.0.3","hardware":{"type":"cuda","name":"RTX 4060","memory_gb":8.0}}' >/dev/null
STATUS=$(curl -s $COORD_URL/status)
NODES=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['num_nodes'])")
PROPS=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['proportions'])")
check "3 nodes registered" "3" "$NODES"
check "3-node proportions computed" "[0.71, 0.05, 0.24]" "$PROPS"

# --- Test 9: Full integration (2 workers + inference) ---
echo ""
echo "--- Full Integration (2 workers) ---"
# Reset coordinator
docker stop rn-coord 2>/dev/null; docker rm rn-coord 2>/dev/null
docker run -d --name rn-coord --network ravnest-test-net -p 8080:8080 \
  ravnest-test python ravnest/cli.py coordinator --port 8080 --min-nodes 2 \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cpu >/dev/null
sleep 3

# Start 2 workers
docker run -d --name rn-w0 --network ravnest-test-net -p 8000:8000 \
  -v ravnest-cache:/tmp/ravnest_model_cache -e PYTHONUNBUFFERED=1 \
  ravnest-test python -c "
import sys; sys.path.insert(0, '/app')
from ravnest.worker import Worker
w = Worker('http://rn-coord:8080', model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0', device='cpu')
w.run()
" >/dev/null

docker run -d --name rn-w1 --network ravnest-test-net \
  -v ravnest-cache:/tmp/ravnest_model_cache -e PYTHONUNBUFFERED=1 \
  ravnest-test python -c "
import sys; sys.path.insert(0, '/app')
from ravnest.worker import Worker
w = Worker('http://rn-coord:8080', model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0', device='cpu')
w.run()
" >/dev/null

echo "Waiting 120s for workers to initialize..."
sleep 120

RESPONSE=$(curl -s -m 300 -X POST $API_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ravnest","messages":[{"role":"user","content":"Hi"}],"max_tokens":3}')
HAS_CONTENT=$(echo "$RESPONSE" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    print('yes' if d['choices'][0]['message']['content'] else 'no')
except: print('no')
" 2>/dev/null)
check "Worker inference returns content" "yes" "$HAS_CONTENT"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
