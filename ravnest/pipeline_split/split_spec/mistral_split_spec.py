from .llama_split_spec import LlamaSplitSpec, LlamaForCausalLMSplitSpec


class MistralSplitSpec(LlamaSplitSpec):
    """Mistral uses the same architecture as Llama (embed_tokens, layers, norm, rotary_emb)."""

    def get_stage_layers(self):
        if self.model.__class__.__name__ == "MistralModel":
            module = self.model
        else:
            module = self.model.model
        return self._get_layers_from_module(module)

    def _get_layers_from_module(self, module):
        from ...strings import NodeTypes
        held_layers = []
        if self.node_type == NodeTypes.ROOT:
            held_layers.append(module.embed_tokens)
        start_idx, end_idx = self.get_stage_layer_indices(len(module.layers))
        held_layers.extend(module.layers[start_idx:end_idx])
        if self.node_type == NodeTypes.LEAF:
            held_layers.append(module.norm)
        if hasattr(module, 'rotary_emb'):
            held_layers.append(module.rotary_emb)
        return held_layers


class MistralForCausalLMSplitSpec(MistralSplitSpec):
    def get_stage_layers(self):
        held_layers = super().get_stage_layers()
        from ...strings import NodeTypes
        if self.node_type == NodeTypes.LEAF:
            held_layers.append(self.model.lm_head)
            if self.tie_weight_check():
                held_layers.append(self.model.model.embed_tokens)
        return held_layers

    def get_shared_params(self):
        if self.tie_weight_check():
            return {
                0: self.model.model.embed_tokens.weight,
                self.num_stages - 1: self.model.lm_head.weight
            }
        return None

    def tie_weight_check(self):
        input_embedding = self.model.get_input_embeddings()
        output_embedding = self.model.get_output_embeddings()
        return (
            input_embedding is not None
            and output_embedding is not None
            and id(input_embedding.weight) == id(output_embedding.weight)
        )
