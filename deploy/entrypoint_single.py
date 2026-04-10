"""
Single-node entrypoint — runs the full model on one machine with no splitting.

Provides an OpenAI-compatible API without any distributed infrastructure.
Useful as a baseline or when you just want a local LLM API server.
"""

import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
if os.path.isdir("/app") and os.path.isdir("/app/ravnest"):
    sys.path.insert(0, "/app")
else:
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _repo_root)


class SingleNodeEngine:
    """Minimal inference engine that wraps a single HuggingFace model.

    Mimics the InferenceEngine interface so the API server works unchanged.
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt_list=None, max_seq_lengths=None, top_k=1, temperature=1.0, return_new_tokens_only=False):
        if not prompt_list:
            return [""]
        prompt = prompt_list[0]
        max_new = max_seq_lengths[0] if max_seq_lengths else 128

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=temperature > 0 and top_k > 1,
                top_k=top_k if top_k > 1 else None,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Slice off the prompt tokens — string-based stripping is unreliable
        # because chat-templated prompts don't byte-match after encode→decode
        new_tokens = output_ids[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        # Return just the new generation (api server will not need to strip)
        return [text]

    def generate_stream(self, prompt_list=None, max_seq_lengths=None, top_k=1, temperature=1.0):
        """Token-by-token streaming using HuggingFace TextIteratorStreamer."""
        if not prompt_list:
            return
        prompt = prompt_list[0]
        max_new = max_seq_lengths[0] if max_seq_lengths else 128

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        try:
            from transformers import TextIteratorStreamer
            import threading

            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new,
                do_sample=temperature > 0 and top_k > 1,
                top_k=top_k if top_k > 1 else None,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )
            thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            for text in streamer:
                if text:
                    yield text
            thread.join()
        except ImportError:
            # Fallback: non-streaming, yield all at once
            result = self.generate(prompt_list, max_seq_lengths, top_k, temperature)
            text = result[0]
            if text.startswith(prompt):
                text = text[len(prompt):]
            yield text


def main():
    model_name = os.environ.get("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    default_cache = "/app/model_cache" if os.path.isdir("/app") else os.path.expanduser("~/.cache/ravnest/models")
    cache_dir = os.environ.get("MODEL_CACHE_DIR", default_cache)
    os.makedirs(cache_dir, exist_ok=True)

    def _default_device():
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    device_str = os.environ.get("RAVNEST_DEVICE", _default_device())
    device = torch.device(device_str)
    use_cpu = device_str == "cpu"

    print(f"[single-node] Model: {model_name}, Device: {device_str}")

    print("[single-node] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    print("[single-node] Loading model...")
    dtype = torch.float32 if use_cpu else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_str if device_str != "mps" else "cpu",
        cache_dir=cache_dir,
    )
    if device_str == "mps":
        model = model.to(device)
    model.eval()

    print(f"[single-node] Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")

    engine = SingleNodeEngine(model, tokenizer, device)

    from deploy.api_server import create_app
    app = create_app(engine, tokenizer)

    import uvicorn
    api_port = int(os.environ.get("RAVNEST_API_PORT", "8000"))
    print(f"[single-node] API server on http://0.0.0.0:{api_port}")
    uvicorn.run(app, host="0.0.0.0", port=api_port, log_level="info")


if __name__ == "__main__":
    main()
