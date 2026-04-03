"""Tests for proportional layer splitting."""
import pytest
import numpy as np
from ravnest.pipeline_split.split_spec.base_split_spec import BaseSplitSpec


class FakeModel:
    pass


class TestGetLayersPerStage:
    def test_equal_split_2_stages(self):
        spec = BaseSplitSpec(stage=0, node_type=None, model=FakeModel(), num_stages=2)
        layers = spec.get_layers_per_stage(22)
        assert layers == [11, 11]
        assert sum(layers) == 22

    def test_equal_split_3_stages(self):
        spec = BaseSplitSpec(stage=0, node_type=None, model=FakeModel(), num_stages=3)
        layers = spec.get_layers_per_stage(22)
        assert sum(layers) == 22
        # Each stage should get 7 or 8 layers
        assert all(l in (7, 8) for l in layers)

    def test_equal_split_uneven(self):
        spec = BaseSplitSpec(stage=0, node_type=None, model=FakeModel(), num_stages=3)
        layers = spec.get_layers_per_stage(10)
        assert sum(layers) == 10

    def test_proportional_split(self):
        spec = BaseSplitSpec(stage=0, node_type=None, model=FakeModel(), num_stages=2, proportions=[0.3, 0.7])
        layers = spec.get_layers_per_stage(22)
        assert sum(layers) == 22
        assert layers[0] < layers[1]
        # 30% of 22 = 6.6, so 7. 70% = 15.4, so 15.
        assert layers[0] in (6, 7)
        assert layers[1] in (15, 16)

    def test_proportional_split_3_stages(self):
        spec = BaseSplitSpec(stage=0, node_type=None, model=FakeModel(), num_stages=3, proportions=[0.5, 0.3, 0.2])
        layers = spec.get_layers_per_stage(22)
        assert sum(layers) == 22
        assert layers[0] > layers[1] > layers[2]

    def test_proportional_minimum_1_layer(self):
        # Even with tiny proportion, each stage gets at least 1 layer
        spec = BaseSplitSpec(stage=0, node_type=None, model=FakeModel(), num_stages=3, proportions=[0.98, 0.01, 0.01])
        layers = spec.get_layers_per_stage(22)
        assert sum(layers) == 22
        assert all(l >= 1 for l in layers)

    def test_proportional_equal(self):
        spec = BaseSplitSpec(stage=0, node_type=None, model=FakeModel(), num_stages=2, proportions=[0.5, 0.5])
        layers = spec.get_layers_per_stage(22)
        assert layers == [11, 11]

    def test_proportional_preserves_total(self):
        for num_layers in [10, 22, 32, 48]:
            for props in [[0.3, 0.7], [0.2, 0.3, 0.5], [0.25, 0.25, 0.25, 0.25]]:
                spec = BaseSplitSpec(stage=0, node_type=None, model=FakeModel(),
                                   num_stages=len(props), proportions=props)
                layers = spec.get_layers_per_stage(num_layers)
                assert sum(layers) == num_layers, f"Failed for {num_layers} layers, props={props}"


class TestGetStageLayerIndices:
    def test_stage_0_gets_first_layers(self):
        spec = BaseSplitSpec(stage=0, node_type=None, model=FakeModel(), num_stages=2)
        start, end = spec.get_stage_layer_indices(22)
        assert start == 0
        assert end == 11

    def test_stage_1_gets_last_layers(self):
        spec = BaseSplitSpec(stage=1, node_type=None, model=FakeModel(), num_stages=2)
        start, end = spec.get_stage_layer_indices(22)
        assert start == 11
        assert end == 22

    def test_proportional_indices(self):
        spec0 = BaseSplitSpec(stage=0, node_type=None, model=FakeModel(), num_stages=2, proportions=[0.3, 0.7])
        start0, end0 = spec0.get_stage_layer_indices(22)

        spec1 = BaseSplitSpec(stage=1, node_type=None, model=FakeModel(), num_stages=2, proportions=[0.3, 0.7])
        start1, end1 = spec1.get_stage_layer_indices(22)

        # No gaps or overlaps
        assert end0 == start1
        assert start0 == 0
        assert end1 == 22

    def test_3_stage_no_gaps(self):
        indices = []
        for stage in range(3):
            spec = BaseSplitSpec(stage=stage, node_type=None, model=FakeModel(), num_stages=3, proportions=[0.5, 0.3, 0.2])
            start, end = spec.get_stage_layer_indices(22)
            indices.append((start, end))

        # No gaps
        assert indices[0][0] == 0
        assert indices[0][1] == indices[1][0]
        assert indices[1][1] == indices[2][0]
        assert indices[2][1] == 22
