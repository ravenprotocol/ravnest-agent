"""
OpenAI-compatible chat completions API server for Ravnest distributed inference.

Supports both streaming (SSE) and non-streaming responses.
Single-request-at-a-time. Wraps InferenceEngine.generate() and generate_stream().

Auth: set RAVNEST_API_KEY env var to require Bearer token auth.
If not set, all requests are allowed (open access).
"""

import json
import os
import time
import threading
import uuid
from typing import List, Optional

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "ravnest"
    messages: List[ChatMessage]
    max_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 1
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class CompletionRequest(BaseModel):
    """Legacy /v1/completions request — single prompt, not chat messages."""
    model: str = "ravnest"
    prompt: str
    max_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 1
    stream: bool = False


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


def create_app(engine, tokenizer):
    app = FastAPI(title="Ravnest Inference API")
    # Inference lock + FIFO queue: requests wait their turn instead of getting 503
    lock = threading.Lock()
    queue_depth = threading.Semaphore(0)  # not used for blocking; lock handles serialization
    queue_size = [0]  # mutable counter; protected by queue_lock
    queue_lock = threading.Lock()
    MAX_QUEUE = int(os.environ.get("RAVNEST_MAX_QUEUE", "16"))
    QUEUE_TIMEOUT = float(os.environ.get("RAVNEST_QUEUE_TIMEOUT", "300"))  # seconds
    MAX_SEQ_LENGTH = 3000
    API_KEY = os.environ.get("RAVNEST_API_KEY", "")

    # CORS — allow browser-based clients (Open WebUI in browser, custom UIs, etc.)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mutable container so engine/tokenizer can be swapped during hot-reconfigure
    app.state.engine = engine
    app.state.tokenizer = tokenizer
    app.state.ready = True  # set to False during model loading/reconfigure
    app.state.model_id = os.environ.get("MODEL_NAME", "ravnest")

    if API_KEY:
        print(f"[api] API key auth enabled (key length: {len(API_KEY)})")
    else:
        print("[api] API key auth disabled (set RAVNEST_API_KEY to enable)")

    def check_auth(request: Request):
        if not API_KEY:
            return
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Use: Authorization: Bearer <your-key>",
            )
        token = auth_header[7:]
        if token != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def build_prompt(messages):
        """Build a prompt using the model's chat template if available.

        Falls back to a generic User:/Assistant: format for non-chat models.
        """
        tokenizer = app.state.tokenizer
        msgs = [{"role": m.role, "content": m.content} for m in messages]

        # Try the tokenizer's chat template first (correct for chat models)
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass  # fall through to generic format

        # Fallback for base models without a chat template
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    def validate_request(request):
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages array is required and must not be empty")

        prompt = build_prompt(request.messages)
        prompt_token_count = len(app.state.tokenizer.encode(prompt))
        total_seq_length = prompt_token_count + request.max_tokens

        if total_seq_length > MAX_SEQ_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"prompt ({prompt_token_count} tokens) + max_tokens ({request.max_tokens}) "
                       f"exceeds max sequence length ({MAX_SEQ_LENGTH})"
            )
        return prompt, prompt_token_count

    # Load the chat UI HTML once at startup
    chat_ui_path = Path(__file__).parent / "chat_ui.html"
    chat_ui_html = chat_ui_path.read_text() if chat_ui_path.exists() else None

    @app.get("/", response_class=HTMLResponse)
    def chat_ui():
        """Built-in web chat UI — open in a browser to chat with the model."""
        if chat_ui_html is None:
            return HTMLResponse(
                "<h1>Ravnest API</h1><p>Chat UI not bundled. POST to /v1/chat/completions.</p>",
                status_code=200,
            )
        return HTMLResponse(chat_ui_html)

    @app.get("/health")
    def health():
        if not app.state.ready:
            return {"status": "loading", "detail": "Model is still loading, try again shortly"}
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models():
        """OpenAI-compatible models endpoint. Returns the loaded model.

        OpenAI clients (Open WebUI, LangChain, etc.) call this on connect
        to discover available models. Returning a single entry — the model
        currently loaded in the cluster.
        """
        return {
            "object": "list",
            "data": [
                {
                    "id": app.state.model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ravnest",
                }
            ],
        }

    def _enqueue_and_acquire():
        """Reserve a queue slot, then block until the lock is acquired.

        Returns nothing on success; raises HTTPException on overflow or timeout.
        """
        with queue_lock:
            if queue_size[0] >= MAX_QUEUE:
                raise HTTPException(
                    status_code=503,
                    detail=f"Queue full ({MAX_QUEUE} requests waiting). Try again later.",
                )
            queue_size[0] += 1

        try:
            acquired = lock.acquire(timeout=QUEUE_TIMEOUT)
            if not acquired:
                raise HTTPException(
                    status_code=504,
                    detail=f"Request timed out after {QUEUE_TIMEOUT}s waiting in queue.",
                )
        except BaseException:
            with queue_lock:
                queue_size[0] -= 1
            raise

    def _release_queue_slot():
        with queue_lock:
            queue_size[0] = max(0, queue_size[0] - 1)

    @app.get("/v1/queue")
    def queue_status():
        """Show the current request queue depth — useful for monitoring."""
        with queue_lock:
            depth = queue_size[0]
        return {"queued": depth, "max": MAX_QUEUE, "ready": app.state.ready}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest, raw_request: Request):
        check_auth(raw_request)
        if not app.state.ready:
            raise HTTPException(
                status_code=503,
                detail="Model is still loading. Check GET /health for status.",
            )
        prompt, prompt_token_count = validate_request(request)

        _enqueue_and_acquire()

        try:
            if request.stream:
                return _stream_response(request, prompt, prompt_token_count)
            else:
                return _non_stream_response(request, prompt, prompt_token_count)
        except HTTPException:
            # Lock already released by _non_stream_response or _stream_response
            raise
        except Exception:
            # Safety net: release lock and queue slot if somehow not released
            try:
                lock.release()
            except RuntimeError:
                pass
            _release_queue_slot()
            raise

    def _non_stream_response(request, prompt, prompt_token_count):
        try:
            start_time = time.time()

            outputs = app.state.engine.generate(
                prompt_list=[prompt],
                max_seq_lengths=[request.max_tokens],
                top_k=request.top_k,
                temperature=request.temperature,
                return_new_tokens_only=True,
            )

            elapsed = time.time() - start_time
            generated_text = (outputs[0] if outputs else "").strip()

            completion_tokens = len(app.state.tokenizer.encode(generated_text))

            print(f"[api] Request completed in {elapsed:.2f}s, "
                  f"prompt={prompt_token_count} tokens, completion={completion_tokens} tokens")

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=generated_text),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_token_count,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_token_count + completion_tokens,
                ),
            )
        except (ConnectionError, BrokenPipeError, OSError) as e:
            raise HTTPException(
                status_code=503,
                detail=f"A node disconnected. Retry in a few seconds. ({e})"
            )
        except RuntimeError as e:
            err = str(e).lower()
            if any(x in err for x in ["connection", "timed out", "broken pipe", "reset"]):
                raise HTTPException(
                    status_code=503,
                    detail=f"A node disconnected. Retry in a few seconds. ({e})"
                )
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
        finally:
            lock.release()
            _release_queue_slot()

    @app.post("/v1/completions")
    def completions(request: CompletionRequest, raw_request: Request):
        """Legacy OpenAI completions endpoint — single prompt, no chat messages.

        Some older clients (text-completion-only tools) use this instead of
        /v1/chat/completions. We treat the prompt as a literal string, no
        chat templating.
        """
        check_auth(raw_request)
        if not app.state.ready:
            raise HTTPException(
                status_code=503,
                detail="Model is still loading. Check GET /health for status.",
            )
        if not request.prompt:
            raise HTTPException(status_code=400, detail="prompt must not be empty")

        prompt = request.prompt
        prompt_token_count = len(app.state.tokenizer.encode(prompt))
        if prompt_token_count + request.max_tokens > MAX_SEQ_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"prompt + max_tokens exceeds {MAX_SEQ_LENGTH}",
            )

        _enqueue_and_acquire()

        try:
            outputs = app.state.engine.generate(
                prompt_list=[prompt],
                max_seq_lengths=[request.max_tokens],
                top_k=request.top_k,
                temperature=request.temperature,
                return_new_tokens_only=True,
            )
            generated_text = outputs[0] if outputs else ""

            completion_tokens = len(app.state.tokenizer.encode(generated_text))

            return CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex[:12]}",
                created=int(time.time()),
                model=request.model,
                choices=[CompletionChoice(text=generated_text, index=0, finish_reason="stop")],
                usage=Usage(
                    prompt_tokens=prompt_token_count,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_token_count + completion_tokens,
                ),
            )
        except (ConnectionError, BrokenPipeError, OSError) as e:
            raise HTTPException(status_code=503, detail=f"A node disconnected. ({e})")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
        finally:
            lock.release()
            _release_queue_slot()

    def _stream_response(request, prompt, prompt_token_count):
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def event_stream():
            try:
                completion_tokens = 0
                for token_text in app.state.engine.generate_stream(
                    prompt_list=[prompt],
                    max_seq_lengths=[request.max_tokens],
                    top_k=request.top_k,
                    temperature=request.temperature,
                ):
                    if not token_text:
                        continue
                    completion_tokens += 1
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

                print(f"[api] Stream completed, "
                      f"prompt={prompt_token_count} tokens, completion={completion_tokens} tokens")
            except Exception as e:
                error_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"\n\n[Error: {str(e)}]"},
                        "finish_reason": "stop",
                    }],
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                lock.release()
                _release_queue_slot()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app
