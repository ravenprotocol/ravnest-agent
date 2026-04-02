from .llama_split_spec import LlamaSplitSpec, LlamaForCausalLMSplitSpec
from .qwen_2_split_spec import Qwen2SplitSpec, Qwen2ForCausalLMSplitSpec
from .mistral_split_spec import MistralSplitSpec, MistralForCausalLMSplitSpec
from .phi_split_spec import Phi3SplitSpec, Phi3ForCausalLMSplitSpec, PhiSplitSpec, PhiForCausalLMSplitSpec

name_spec_mapping = {
    # Llama
    'LlamaModel': LlamaSplitSpec,
    'LlamaForCausalLM': LlamaForCausalLMSplitSpec,
    # Qwen-2
    'Qwen2Model': Qwen2SplitSpec,
    'Qwen2ForCausalLM': Qwen2ForCausalLMSplitSpec,
    # Mistral
    'MistralModel': MistralSplitSpec,
    'MistralForCausalLM': MistralForCausalLMSplitSpec,
    # Phi-3 / Phi-3.5
    'Phi3Model': Phi3SplitSpec,
    'Phi3ForCausalLM': Phi3ForCausalLMSplitSpec,
    # Phi-1 / Phi-2
    'PhiModel': PhiSplitSpec,
    'PhiForCausalLM': PhiForCausalLMSplitSpec,
}

def get_split_spec(model):
    model_class_name = model.__class__.__name__
    return name_spec_mapping.get(model_class_name)