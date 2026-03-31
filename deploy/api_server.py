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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
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


def create_app(engine, tokenizer):
    app = FastAPI(title="Ravnest Inference API")
    lock = threading.Lock()
    MAX_SEQ_LENGTH = 3000
    API_KEY = os.environ.get("RAVNEST_API_KEY", "")

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
        prompt_token_count = len(tokenizer.encode(prompt))
        total_seq_length = prompt_token_count + request.max_tokens

        if total_seq_length > MAX_SEQ_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"prompt ({prompt_token_count} tokens) + max_tokens ({request.max_tokens}) "
                       f"exceeds max sequence length ({MAX_SEQ_LENGTH})"
            )
        return prompt, prompt_token_count

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest, raw_request: Request):
        check_auth(raw_request)
        prompt, prompt_token_count = validate_request(request)

        acquired = lock.acquire(blocking=False)
        if not acquired:
            raise HTTPException(status_code=503, detail="Server busy, try again later")

        try:
            if request.stream:
                return _stream_response(request, prompt, prompt_token_count)
            else:
                return _non_stream_response(request, prompt, prompt_token_count)
        except HTTPException:
            lock.release()
            raise
        except RuntimeError as e:
            lock.release()
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        except Exception as e:
            lock.release()
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    def _non_stream_response(request, prompt, prompt_token_count):
        try:
            start_time = time.time()

            outputs = engine.generate(
                prompt_list=[prompt],
                max_seq_lengths=[request.max_tokens],
                top_k=request.top_k,
                temperature=request.temperature,
            )

            elapsed = time.time() - start_time
            generated_text = outputs[0] if outputs else ""

            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            generated_text = generated_text.strip()

            completion_tokens = len(tokenizer.encode(generated_text))

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
        finally:
            lock.release()

    def _stream_response(request, prompt, prompt_token_count):
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def event_stream():
            try:
                completion_tokens = 0
                for token_text in engine.generate_stream(
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
