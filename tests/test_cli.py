"""Tests for CLI command logic (no Docker or torch needed)."""
import pytest
import os
import sys

# Import CLI functions directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ravnest.cli import generate_compose, detect_hardware, SUPPORTED_MODELS


class TestGenerateCompose:
    def test_basic_2_node(self):
        compose = generate_compose("test-model", nodes=2, device="cpu", port=8000)
        assert "node-0:" in compose
        assert "node-1:" in compose
        assert "RANK=0" in compose
        assert "RANK=1" in compose
        assert "WORLD_SIZE=2" in compose
        assert "test-model" in compose

    def test_gpu_includes_nvidia_runtime(self):
        compose = generate_compose("test-model", nodes=2, device="cuda", port=8000)
        assert "runtime: nvidia" in compose
        assert "capabilities: [gpu]" in compose

    def test_cpu_no_nvidia(self):
        compose = generate_compose("test-model", nodes=2, device="cpu", port=8000)
        assert "nvidia" not in compose

    def test_port_mapping(self):
        compose = generate_compose("test-model", nodes=2, device="cpu", port=9000)
        assert '"9000:8000"' in compose

    def test_api_key(self):
        compose = generate_compose("test-model", nodes=2, device="cpu", port=8000, api_key="secret")
        assert "RAVNEST_API_KEY=secret" in compose

    def test_no_api_key_on_leaf(self):
        compose = generate_compose("test-model", nodes=2, device="cpu", port=8000, api_key="secret")
        # API key should only appear once (on node-0)
        assert compose.count("RAVNEST_API_KEY=secret") == 1

    def test_cross_machine_host_networking(self):
        compose = generate_compose("test-model", nodes=1, device="cpu", port=8000,
                                   master_addr="192.168.1.100", network_mode="host")
        assert "network_mode: host" in compose
        assert "MASTER_ADDR=192.168.1.100" in compose

    def test_proportions(self):
        compose = generate_compose("test-model", nodes=2, device="cpu", port=8000,
                                   proportions=[0.3, 0.7])
        assert "RAVNEST_PROPORTIONS=0.3,0.7" in compose

    def test_auto_profile(self):
        compose = generate_compose("test-model", nodes=2, device="cpu", port=8000,
                                   auto_profile=True)
        assert "RAVNEST_AUTO_PROFILE=true" in compose

    def test_world_size_override(self):
        compose = generate_compose("test-model", nodes=1, device="cpu", port=8000,
                                   world_size=3)
        assert "WORLD_SIZE=3" in compose

    def test_rank_offset(self):
        compose = generate_compose("test-model", nodes=1, device="cpu", port=8000,
                                   rank_offset=2, world_size=3)
        assert "RANK=2" in compose
        assert "NODE_ROLE=leaf" in compose

    def test_volume_present(self):
        compose = generate_compose("test-model", nodes=2, device="cpu", port=8000)
        assert "model_cache" in compose

    def test_depends_on(self):
        compose = generate_compose("test-model", nodes=2, device="cpu", port=8000)
        assert "depends_on:" in compose
        assert "- node-0" in compose


class TestSupportedModels:
    def test_models_list_not_empty(self):
        assert len(SUPPORTED_MODELS) > 0

    def test_each_model_has_required_fields(self):
        for m in SUPPORTED_MODELS:
            assert "id" in m
            assert "arch" in m
            assert "params" in m
            assert "size" in m
            assert "min_ram" in m
            assert "gated" in m

    def test_tinyllama_is_not_gated(self):
        tiny = [m for m in SUPPORTED_MODELS if "TinyLlama" in m["id"]]
        assert len(tiny) == 1
        assert tiny[0]["gated"] is False

    def test_llama_models_are_gated(self):
        llama = [m for m in SUPPORTED_MODELS if "meta-llama" in m["id"]]
        assert all(m["gated"] for m in llama)

    def test_architectures_covered(self):
        archs = set(m["arch"] for m in SUPPORTED_MODELS)
        assert "Llama" in archs
        assert "Mistral" in archs
        assert "Qwen-2" in archs


class TestDetectHardware:
    def test_returns_dict(self):
        hw = detect_hardware()
        assert isinstance(hw, dict)
        assert "device" in hw
        assert "gpu_count" in hw
        assert "gpus" in hw

    def test_device_is_valid(self):
        hw = detect_hardware()
        assert hw["device"] in ("cpu", "cuda")
