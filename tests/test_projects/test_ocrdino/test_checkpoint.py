"""Checkpoint migration contract; these tests do not need PyTorch."""
import importlib.util
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / 'projects/OCRDINO/tools/convert_checkpoint.py'
SPEC = importlib.util.spec_from_file_location('ocr_checkpoint_conversion', TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TensorShape:
    def __init__(self, *shape):
        self.shape = shape


def official_state():
    state = {
        'query_embedding.weight': TensorShape(900, 256),
        'level_embed': TensorShape(4, 256),
        'backbone.conv1.weight': TensorShape(64, 3, 7, 7),
        'dn_query_generator.label_embedding.weight': TensorShape(80, 256),
        'encoder.layers.0.ffn.weight': TensorShape(256, 256),
        'decoder.layers.0.ffn.weight': TensorShape(256, 256),
        'bbox_head.reg_branches.0.weight': TensorShape(4, 256)}
    for layer in range(7):
        state['bbox_head.cls_branches.{}.weight'.format(layer)] = TensorShape(80, 256)
        state['bbox_head.cls_branches.{}.bias'.format(layer)] = TensorShape(80)
    return state


class CheckpointMigrationTest(unittest.TestCase):
    def test_preserves_localization_transformer_backbone_by_identity(self):
        source = official_state()
        output, report = MODULE.convert_state_dict(source)
        self.assertEqual(len(report['dropped_tensors']), 15)
        self.assertEqual(len(source), len(output) + 15)
        for name, tensor in output.items():
            self.assertIs(tensor, source[name])
        self.assertIn('dn_query_generator.label_embedding.weight', source)
        self.assertIn('three Character Query decoder layers', report['newly_initialized'])
        self.assertIn('character plus EOS classifier', report['newly_initialized'])
        self.assertFalse(any('CTC' in item or 'BiLSTM' in item
                             for item in report['newly_initialized']))

    def test_distributed_prefix_does_not_lose_other_weights(self):
        source = official_state()
        output, _ = MODULE.convert_state_dict(
            {'module.' + key: value for key, value in source.items()})
        self.assertIs(output['query_embedding.weight'], source['query_embedding.weight'])

    def test_rejects_wrong_backbone_scale_query_or_class_count(self):
        for name, tensor in [
                ('level_embed', TensorShape(5, 256)),
                ('query_embedding.weight', TensorShape(300, 256)),
                ('dn_query_generator.label_embedding.weight', TensorShape(1, 256))]:
            state = official_state()
            state[name] = tensor
            with self.subTest(name=name), self.assertRaises(ValueError):
                MODULE.convert_state_dict(state)

    def test_rejects_existing_ocr_head_or_missing_class_branches(self):
        state = official_state()
        state['recognition_head.classifier.weight'] = TensorShape(96, 256)
        with self.assertRaises(ValueError):
            MODULE.convert_state_dict(state)
        state = official_state()
        state.pop('bbox_head.cls_branches.6.bias')
        with self.assertRaises(ValueError):
            MODULE.convert_state_dict(state)


if __name__ == '__main__':
    unittest.main()
