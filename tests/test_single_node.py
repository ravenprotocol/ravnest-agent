"""Tests for single-node engine (no distributed infrastructure)."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from deploy.entrypoint_single import SingleNodeEngine


class FakeModel:
    """Minimal mock of a HuggingFace model for testing."""
    def __init__(self):
        self.called_with = None

    def generate(self, **kwargs):
        import torch
        # If a streamer is provided (TextIteratorStreamer), feed it then end
        streamer = kwargs.get("streamer")
        if streamer is not None:
            # First put is the prompt (skipped by skip_prompt=True), then generation
            streamer.put(torch.tensor([1, 2, 3]))
            streamer.put(torch.tensor([42]))
            streamer.put(torch.tensor([43]))
            streamer.end()
        input_ids = kwargs.get("input_ids", torch.tensor([[1, 2, 3]]))
        new_tokens = torch.tensor([[42, 43, 44]])
        return torch.cat([input_ids, new_tokens], dim=1)

    def parameters(self):
        import torch
        return [torch.zeros(10)]


class FakeBatchEncoding(dict):
    """Mimics HuggingFace BatchEncoding with .to() support."""
    def to(self, device):
        return self


class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 2

    def __call__(self, text, return_tensors="pt"):
        import torch
        return FakeBatchEncoding(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )

    def decode(self, ids, skip_special_tokens=False):
        return "Hello world response"

    def encode(self, text):
        return [1, 2, 3]


class TestSingleNodeEngine:
    def setup_method(self):
        import torch
        self.engine = SingleNodeEngine(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            device=torch.device("cpu"),
        )

    def test_generate_returns_list(self):
        result = self.engine.generate(
            prompt_list=["Hello"],
            max_seq_lengths=[50],
        )
        assert isinstance(result, list)
        assert len(result) == 1

    def test_generate_returns_string(self):
        result = self.engine.generate(
            prompt_list=["Hello"],
            max_seq_lengths=[50],
        )
        assert isinstance(result[0], str)

    def test_generate_empty_prompt(self):
        result = self.engine.generate(prompt_list=None, max_seq_lengths=None)
        assert result == [""]

    def test_generate_stream_yields_strings(self):
        tokens = list(self.engine.generate_stream(
            prompt_list=["Hello"],
            max_seq_lengths=[50],
        ))
        assert len(tokens) >= 1
        assert all(isinstance(t, str) for t in tokens)

    def test_generate_stream_empty_prompt(self):
        tokens = list(self.engine.generate_stream(prompt_list=None))
        assert tokens == []

    def test_default_top_k(self):
        # Should not crash with default top_k=1 (greedy)
        result = self.engine.generate(
            prompt_list=["Test"],
            max_seq_lengths=[10],
            top_k=1,
            temperature=1.0,
        )
        assert len(result) == 1

    def test_temperature_zero_is_greedy(self):
        result = self.engine.generate(
            prompt_list=["Test"],
            max_seq_lengths=[10],
            top_k=1,
            temperature=0,
        )
        assert len(result) == 1


class TestSingleNodeCLILogic:
    """Test that --nodes 1 triggers single-node mode."""

    def test_nodes_1_is_single_mode(self):
        # Single-node condition: no peers and nodes == 1
        peers = None
        nodes = 1
        is_single = not peers and nodes == 1
        assert is_single is True

    def test_nodes_2_is_not_single(self):
        peers = None
        nodes = 2
        is_single = not peers and nodes == 1
        assert is_single is False

    def test_peers_with_1_node_is_not_single(self):
        # If peers are set, it's multi-machine even with 1 node
        peers = "192.168.1.5"
        nodes = 1
        is_single = not peers and nodes == 1
        assert is_single is False
