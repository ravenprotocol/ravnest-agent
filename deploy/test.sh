#!/bin/bash
# Smoke test for Ravnest distributed inference demo
set -e

API_URL="${API_URL:-http://localhost:8000}"
API_KEY="${RAVNEST_API_KEY:-}"
PASS=0
FAIL=0

# Build auth args for curl
AUTH=()
if [ -n "$API_KEY" ]; then
    AUTH=(-H "Authorization: Bearer $API_KEY")
fi

check() {
    local name="$1"
    local expected_code="$2"
    local actual_code="$3"
    if [ "$actual_code" = "$expected_code" ]; then
        echo "PASS: $name (HTTP $actual_code)"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $name (expected HTTP $expected_code, got $actual_code)"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Ravnest Smoke Tests ==="
echo "API: $API_URL"
[ -n "$API_KEY" ] && echo "Auth: enabled" || echo "Auth: disabled"
echo ""

# 1. Health check (no auth required)
echo "--- Health Check ---"
CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health")
check "Health endpoint" "200" "$CODE"

# 2. Happy path
echo "--- Happy Path ---"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    "${AUTH[@]}" \
    -d '{"model":"ravnest","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":10}')
CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)
check "Chat completion" "200" "$CODE"

if echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['choices'][0]['message']['content']" 2>/dev/null; then
    echo "PASS: Response has content"
    PASS=$((PASS + 1))
else
    echo "FAIL: Response missing content"
    FAIL=$((FAIL + 1))
fi

# 3. Empty messages
echo "--- Empty Messages ---"
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    "${AUTH[@]}" \
    -d '{"model":"ravnest","messages":[],"max_tokens":10}')
check "Empty messages" "400" "$CODE"

# 4. Auth tests (only when API key is set)
if [ -n "$API_KEY" ]; then
    echo "--- Auth Tests ---"
    CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"ravnest","messages":[{"role":"user","content":"Hi"}],"max_tokens":2}')
    check "No auth header rejected" "401" "$CODE"

    CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer wrong-key-here" \
        -d '{"model":"ravnest","messages":[{"role":"user","content":"Hi"}],"max_tokens":2}')
    check "Wrong API key rejected" "401" "$CODE"
fi

# 5. Concurrent request
echo "--- Concurrent Request ---"
curl -s -o /dev/null -X POST "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    "${AUTH[@]}" \
    -d '{"model":"ravnest","messages":[{"role":"user","content":"Write a long story."}],"max_tokens":50}' &
BG_PID=$!
sleep 1
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    "${AUTH[@]}" \
    -d '{"model":"ravnest","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}')
check "Concurrent request rejected" "503" "$CODE"
wait $BG_PID 2>/dev/null || true

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
