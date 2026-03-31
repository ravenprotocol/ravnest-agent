import numpy as np

class BaseSplitSpec():
    def __init__(self, stage, node_type, model, num_stages, proportions=None):
        self.stage = stage
        self.node_type = node_type
        self.model = model
        self.num_stages = num_stages
        self.proportions = proportions  # e.g. [0.3, 0.7] for 2 nodes

    def get_stage_layer_indices(self, num_layers):
        layers_per_stage = self.get_layers_per_stage(num_layers)
        num_layers_per_stage_accumulated = np.insert(np.cumsum(layers_per_stage), 0, 0)

        self.start_idx = num_layers_per_stage_accumulated[self.stage]
        self.end_idx = num_layers_per_stage_accumulated[self.stage + 1]
        return [self.start_idx, self.end_idx]

    def get_layers_per_stage(self, num_layers):
        if self.proportions is not None:
            return self._get_layers_proportional(num_layers)

        quotient = num_layers // self.num_stages
        remainder = num_layers % self.num_stages

        layers_per_stage = [quotient] * self.num_stages
        if remainder > 0:
            start_position = self.num_stages // 2 - remainder // 2
            for i in range(start_position, start_position + remainder):
                layers_per_stage[i] += 1

        return layers_per_stage

    def _get_layers_proportional(self, num_layers):
        """Split layers according to proportions (e.g. [0.3, 0.7])."""
        assert len(self.proportions) == self.num_stages
        raw = [p * num_layers for p in self.proportions]
        # Round down, then distribute remainder to largest fractional parts
        layers_per_stage = [int(r) for r in raw]
        remainder = num_layers - sum(layers_per_stage)
        fractions = [(raw[i] - layers_per_stage[i], i) for i in range(self.num_stages)]
        fractions.sort(reverse=True)
        for j in range(remainder):
            layers_per_stage[fractions[j][1]] += 1
        # Ensure every stage has at least 1 layer
        for i in range(self.num_stages):
            if layers_per_stage[i] == 0:
                # Steal from the largest neighbor
                donor = max(range(self.num_stages), key=lambda x: layers_per_stage[x])
                layers_per_stage[donor] -= 1
                layers_per_stage[i] = 1
        return layers_per_stage
    
    def configure_stage_model(self):
        '''
        Configures model for this pipeline stage by retaining only the required layers
        '''
        ...