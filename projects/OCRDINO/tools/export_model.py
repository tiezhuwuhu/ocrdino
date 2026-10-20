"""Export the final OCRDINO epoch as model weights for inference/fine-tuning.

Run in the PyTorch/MMEngine training environment; no GPU/model build needed.
The source is a trusted MMEngine training checkpoint.
All model parameters and persistent buffers are retained without dtype changes.
Optimizer/scaler/scheduler/progress/logging state is deliberately not exported.
"""
import argparse
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path


def export_checkpoint(source, output=None, expected_epoch=24):
    """Return an export summary; never overwrite the source or an existing file.

    bestepoch.pth means the user-selected final epoch, not a checkpoint
    chosen by validation accuracy. Keep the original file for --resume.
    """
    import torch

    if (isinstance(expected_epoch, bool) or
            not isinstance(expected_epoch, int) or expected_epoch < 1):
        raise ValueError('expected_epoch must be a positive integer')
    source = Path(source).expanduser().resolve()
    output = (source.parent / 'bestepoch.pth' if output is None
              else Path(output).expanduser().resolve())
    if source == output:
        raise ValueError('Source and output must be different; source is read-only')
    if output.exists():
        raise FileExistsError('Output already exists; choose a new path: ' + str(output))
    if not source.is_file():
        raise FileNotFoundError(source)

    source_bytes = source.stat().st_size
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Expected an MMEngine training checkpoint dictionary')
    state = checkpoint.get('state_dict')
    meta = checkpoint.get('meta')
    if not isinstance(state, Mapping) or not state:
        raise ValueError('Checkpoint needs a nonempty state_dict')
    if not isinstance(meta, Mapping):
        raise ValueError('Checkpoint needs meta.epoch to verify the final epoch')
    epoch = meta.get('epoch')
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch != expected_epoch:
        raise ValueError(
            f'Expected completed epoch {expected_epoch}, found meta.epoch={epoch!r}. '
            'Choose the final checkpoint or explicitly set --expected-epoch.')

    has_recognition = any(
        isinstance(name, str) and name.removeprefix('module.').startswith(
            'recognition_head.') for name in state)
    if has_recognition and (
            not isinstance(meta.get('ocrdino'), dict) or not meta['ocrdino']):
        raise ValueError(
            'OCR recognition weights need meta.ocrdino dictionary/version information')

    exported_state = OrderedDict()
    for name, tensor in state.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise ValueError('Expected named tensor model state, got ' + repr(name))
        if (tensor.is_floating_point() or tensor.is_complex()):
            if not torch.isfinite(tensor).all().item():
                raise ValueError(
                    f'Model tensor {name!r} contains NaN/Inf; export cannot repair weights')
        # Own storage prevents a tensor view from serializing a much larger
        # backing allocation. Keep exact values, keys, shape and dtype.
        exported_state[name] = tensor.detach().cpu().clone()
    if hasattr(state, '_metadata'):
        exported_state._metadata = deepcopy(state._metadata)

    # OCR semantics are model identity, not training state. Query checkpoints
    # require these fields when loaded through OCRDictionaryHook.
    exported_meta = {}
    for key in ('ocrdino', 'dataset_meta'):
        if key in meta:
            if not isinstance(meta[key], dict):
                raise ValueError('Model metadata must be a dictionary: ' + key)
            exported_meta[key] = deepcopy(meta[key])
    payload = dict(state_dict=exported_state, meta=exported_meta)
    report = dict(
        source=str(source), output=str(output), epoch=epoch,
        tensor_count=len(exported_state),
        removed_top_level_keys=sorted(set(checkpoint) - {'state_dict', 'meta'}),
        removed_meta_keys=sorted(set(meta) - set(exported_meta)),
        source_bytes=source_bytes)
    # Release Adam/scaler/log state before serialization. Historical NaNs in
    # those discarded fields do not imply the model tensors are corrupt.
    del checkpoint, state, meta
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also refuses an output created between the initial
    # existence check and this write.
    handle = output.open('xb')
    try:
        with handle:
            torch.save(payload, handle)
    except BaseException:
        # Only remove the incomplete new file created by this invocation.
        output.unlink()
        raise
    report['output_bytes'] = output.stat().st_size
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True,
                        help='Trusted full checkpoint, normally epoch_24.pth')
    parser.add_argument('--output',
                        help='Default: bestepoch.pth next to the source')
    parser.add_argument('--expected-epoch', type=int, default=24,
                        help='Verify completed epoch; use 12 for final fine-tuning weights')
    args = parser.parse_args()
    report = export_checkpoint(args.source, args.output, args.expected_epoch)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
