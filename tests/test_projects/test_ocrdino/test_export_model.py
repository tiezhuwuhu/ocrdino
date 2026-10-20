"""Model-only exports preserve OCR identity while dropping training state."""
import hashlib
import importlib.util
import json
from collections import OrderedDict
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / 'projects/OCRDINO/tools/export_model.py'
SPEC = importlib.util.spec_from_file_location('ocrdino_export_model', TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
VOCAB_SPEC = importlib.util.spec_from_file_location(
    'ocrdino_export_vocabulary',
    ROOT / 'projects/OCRDINO/ocrdino/vocabulary.py')
VOCAB = importlib.util.module_from_spec(VOCAB_SPEC)
VOCAB_SPEC.loader.exec_module(VOCAB)


def ocr_metadata():
    dictionary = VOCAB.CharacterDictionary(['A', 'B', 'L'])
    if hasattr(VOCAB, 'recognition_metadata'):
        return VOCAB.recognition_metadata(dictionary)
    return dict(
        architecture='dino_c2_conv4_bilstm1_ctc',
        characters=list(dictionary.chars), blank=0,
        bbox_format='xyxy', dictionary_case_sensitive=True)


def checkpoint(epoch=24):
    state = OrderedDict([
        ('backbone.weight', torch.arange(6, dtype=torch.float32).reshape(2, 3)),
        ('backbone.running_var', torch.tensor([0.25, 0.5], dtype=torch.float16)),
        ('backbone.num_batches_tracked', torch.tensor(17, dtype=torch.int64)),
        ('recognition_head.classifier.weight',
         torch.arange(12, dtype=torch.float64).reshape(4, 3)),
        ('recognition_head.dict_hash',
         torch.tensor([0, 127, 255], dtype=torch.uint8)),
        ('recognition_head.recognition_version', torch.tensor(1)),
        ('backbone.enabled', torch.tensor(True)),
    ])
    state._metadata = OrderedDict([('', {'version': 1}), ('backbone', {'version': 2})])
    return dict(
        state_dict=state,
        meta=dict(epoch=epoch, iter=96, seed=3, cfg='training-only config',
                  ocrdino=ocr_metadata(),
                  dataset_meta=dict(classes=['text'], palette=[[255, 0, 0]])),
        optimizer=dict(state={0: {'exp_avg': torch.zeros(128),
                                  'exp_avg_sq': torch.ones(128)}},
                       param_groups=[{'lr': 1e-4, 'params': [0]}]),
        param_schedulers=[{'last_step': 24}],
        message_hub={'grad_norm': float('nan')},
        loss_scaler={'scale': 64},
        scaler={'scale': 64},
        cfg='top-level training config',
        epoch=epoch, iter=96)


def save_source(tmp_path, payload=None):
    source = tmp_path / 'epoch_24.pth'
    torch.save(checkpoint() if payload is None else payload, source)
    return source


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_export(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def test_default_export_preserves_model_and_discards_training(tmp_path):
    original = checkpoint()
    source = save_source(tmp_path, original)
    source_hash = digest(source)
    report = MODULE.export_checkpoint(source)
    output = tmp_path / 'bestepoch.pth'
    saved = read_export(output)

    assert set(saved) == {'state_dict', 'meta'}
    assert set(saved['meta']) == {'ocrdino', 'dataset_meta'}
    assert saved['meta']['ocrdino'] == original['meta']['ocrdino']
    assert saved['meta']['dataset_meta'] == original['meta']['dataset_meta']
    assert list(saved['state_dict']) == list(original['state_dict'])
    assert saved['state_dict']._metadata == original['state_dict']._metadata
    for name, expected in original['state_dict'].items():
        actual = saved['state_dict'][name]
        assert actual.device.type == 'cpu'
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
    assert digest(source) == source_hash
    assert report['tensor_count'] == len(original['state_dict'])
    assert report['epoch'] == 24
    assert Path(report['source']) == source.resolve()
    assert Path(report['output']) == output.resolve()
    assert report['source_bytes'] == source.stat().st_size
    assert report['output_bytes'] == output.stat().st_size
    assert set(report['removed_top_level_keys']) == (
        set(original) - {'state_dict', 'meta'})
    assert set(report['removed_meta_keys']) == {'epoch', 'iter', 'seed', 'cfg'}


def test_exported_identity_is_accepted_by_current_recognizer(tmp_path):
    source = save_source(tmp_path)
    saved = read_export(MODULE.export_checkpoint(source)['output'])
    dictionary = VOCAB.CharacterDictionary(['A', 'B', 'L'])
    if hasattr(VOCAB, 'validate_checkpoint_metadata'):
        VOCAB.validate_checkpoint_metadata(saved, dictionary)
        assert saved['meta']['ocrdino']['eos_id'] == 3
        assert saved['meta']['ocrdino']['dictionary_hash'] == dictionary.hash_hex
        assert 'blank' not in saved['meta']['ocrdino']
    else:
        assert saved['meta']['ocrdino']['blank'] == dictionary.blank_idx
        assert saved['meta']['ocrdino']['characters'] == list(dictionary.chars)


def test_explicit_final_finetune_epoch_and_nested_output(tmp_path):
    source = save_source(tmp_path, checkpoint(epoch=12))
    output = tmp_path / 'exports' / 'totaltext_final.pth'
    report = MODULE.export_checkpoint(source, output, expected_epoch=12)
    assert report['epoch'] == 12
    assert output.is_file()
    assert 'epoch' not in read_export(output)['meta']


def test_output_collision_preserves_both_files(tmp_path):
    source = save_source(tmp_path)
    output = tmp_path / 'bestepoch.pth'
    output.write_bytes(b'existing output must be preserved')
    source_hash, output_hash = digest(source), digest(output)
    with pytest.raises(FileExistsError):
        MODULE.export_checkpoint(source)
    assert digest(source) == source_hash
    assert digest(output) == output_hash


def test_source_cannot_be_used_as_output(tmp_path):
    source = save_source(tmp_path)
    before = digest(source)
    with pytest.raises(ValueError, match='different'):
        MODULE.export_checkpoint(source, source)
    assert digest(source) == before


@pytest.mark.parametrize('epoch', [16, 25, '24', True, None])
def test_wrong_or_malformed_epoch_is_rejected(tmp_path, epoch):
    source = save_source(tmp_path, checkpoint(epoch=epoch))
    with pytest.raises(ValueError, match='epoch'):
        MODULE.export_checkpoint(source)
    assert not (tmp_path / 'bestepoch.pth').exists()


@pytest.mark.parametrize('expected', [0, -1, True, 24.0])
def test_expected_epoch_must_be_positive_integer(tmp_path, expected):
    source = save_source(tmp_path)
    with pytest.raises(ValueError, match='positive integer'):
        MODULE.export_checkpoint(source, expected_epoch=expected)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_model_tensor_is_rejected_without_output(tmp_path, value):
    payload = checkpoint()
    payload['state_dict']['backbone.weight'][0, 0] = value
    source = save_source(tmp_path, payload)
    before = digest(source)
    with pytest.raises(ValueError, match='NaN/Inf'):
        MODULE.export_checkpoint(source)
    assert digest(source) == before
    assert not (tmp_path / 'bestepoch.pth').exists()


def test_nonfinite_optimizer_and_logging_state_are_discarded(tmp_path):
    payload = checkpoint()
    payload['optimizer']['state'][0]['exp_avg'][0] = float('nan')
    payload['optimizer']['state'][0]['exp_avg_sq'][0] = float('inf')
    source = save_source(tmp_path, payload)
    saved = read_export(MODULE.export_checkpoint(source)['output'])
    assert set(saved) == {'state_dict', 'meta'}
    assert all(torch.isfinite(t).all() for t in saved['state_dict'].values())


@pytest.mark.parametrize('metadata', [None, {}, 'invalid metadata'])
def test_recognition_weights_need_model_identity(tmp_path, metadata):
    payload = checkpoint()
    if metadata is None:
        payload['meta'].pop('ocrdino')
    else:
        payload['meta']['ocrdino'] = metadata
    source = save_source(tmp_path, payload)
    with pytest.raises(ValueError, match='meta.ocrdino'):
        MODULE.export_checkpoint(source)
    assert not (tmp_path / 'bestepoch.pth').exists()


def test_module_prefix_is_kept_and_requires_identity(tmp_path):
    payload = checkpoint()
    payload['state_dict'] = OrderedDict(
        ('module.' + name, tensor) for name, tensor in payload['state_dict'].items())
    source = save_source(tmp_path, payload)
    output = MODULE.export_checkpoint(source)['output']
    assert list(read_export(output)['state_dict']) == list(payload['state_dict'])
    payload['meta'].pop('ocrdino')
    second_source = tmp_path / 'missing_identity.pth'
    torch.save(payload, second_source)
    with pytest.raises(ValueError, match='meta.ocrdino'):
        MODULE.export_checkpoint(second_source, tmp_path / 'invalid.pth')


def test_detection_only_state_does_not_require_ocr_metadata(tmp_path):
    payload = checkpoint()
    payload['state_dict'] = {'backbone.weight': torch.ones(2)}
    payload['meta'].pop('ocrdino')
    source = save_source(tmp_path, payload)
    saved = read_export(MODULE.export_checkpoint(source)['output'])
    assert set(saved['meta']) == {'dataset_meta'}


def test_non_tensor_state_is_rejected(tmp_path):
    payload = checkpoint()
    payload['state_dict']['unexpected'] = 'not a model tensor'
    source = save_source(tmp_path, payload)
    with pytest.raises(ValueError, match='named tensor'):
        MODULE.export_checkpoint(source)


def test_failed_serialization_removes_only_incomplete_output(tmp_path, monkeypatch):
    source = save_source(tmp_path)
    before = digest(source)

    def fail_save(payload, handle):
        handle.write(b'incomplete')
        raise OSError('synthetic disk write failure')

    monkeypatch.setattr(torch, 'save', fail_save)
    with pytest.raises(OSError, match='synthetic disk'):
        MODULE.export_checkpoint(source)
    assert not (tmp_path / 'bestepoch.pth').exists()
    assert digest(source) == before


def test_cli_exports_and_prints_machine_readable_report(tmp_path):
    source = save_source(tmp_path)
    output = tmp_path / 'cli_final.pth'
    run = subprocess.run(
        [sys.executable, '-B', str(TOOL), '--source', str(source),
         '--output', str(output), '--expected-epoch', '24'],
        capture_output=True, text=True, encoding='utf-8', check=True)
    report = json.loads(run.stdout)
    assert Path(report['output']) == output.resolve()
    assert report['epoch'] == 24
    assert 'optimizer' not in read_export(output)


def test_real_mmengine_history_objects_are_removed(tmp_path):
    from mmengine.logging import HistoryBuffer

    payload = checkpoint()
    history = HistoryBuffer()
    history.update(0.5)
    history.update(float('nan'))
    payload['message_hub'] = dict(
        log_scalars={'train/grad_norm': history},
        runtime_info={'epoch': 24, 'iter': 96})
    source = save_source(tmp_path, payload)
    before = digest(source)
    output = MODULE.export_checkpoint(source)['output']
    saved = read_export(output)
    assert set(saved) == {'state_dict', 'meta'}
    assert 'message_hub' not in saved
    assert digest(source) == before
