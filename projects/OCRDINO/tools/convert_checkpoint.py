"""Convert the supplied MMDetection COCO DINO checkpoint without overwriting it."""
import argparse
import json
from pathlib import Path


def convert_state_dict(state):
    """Drop only category-specific COCO tensors; reject unrelated architectures."""
    cleaned = {}
    for name, tensor in state.items():
        name = name.removeprefix('module.')
        if name in cleaned:
            raise ValueError('Duplicate checkpoint key after prefix removal: ' + name)
        cleaned[name] = tensor
    for name, shape in (
            ('query_embedding.weight', (900, 256)), ('level_embed', (4, 256)),
            ('backbone.conv1.weight', (64, 3, 7, 7)),
            ('dn_query_generator.label_embedding.weight', (80, 256))):
        if name not in cleaned or tuple(cleaned[name].shape) != shape:
            raise ValueError('Expected official R50-4scale COCO DINO tensor {} {}'.format(name, shape))
    if not any(name.startswith('encoder.layers.') for name in cleaned):
        raise ValueError('Missing DINO encoder')
    if not any(name.startswith('decoder.layers.') for name in cleaned):
        raise ValueError('Missing DINO decoder')
    if any(name.startswith('recognition_head.') for name in cleaned):
        raise ValueError('Use the official COCO checkpoint, not an OCR checkpoint')
    dropped, retained = {}, {}
    for name, tensor in cleaned.items():
        if name.startswith('bbox_head.cls_branches.') or name == 'dn_query_generator.label_embedding.weight':
            dropped[name] = list(tensor.shape)
        else:
            retained[name] = tensor
    if len([key for key in dropped if key.startswith('bbox_head.cls_branches.')]) != 14:
        raise ValueError('Expected seven COCO classification branches (weight and bias)')
    return retained, dict(
        retained_tensor_count=len(retained), dropped_tensors=dropped,
        newly_initialized=['text classification branches', 'DN text label embedding',
                           'C2 projection', 'four recognition convolutions',
                           'single BiLSTM', 'CTC classifier'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report')
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    report_path = Path(args.report).resolve() if args.report else output.with_suffix('.report.json')
    if source in (output, report_path) or output == report_path:
        parser.error('Source, output and report paths must be distinct')
    if output.exists() or report_path.exists():
        parser.error('Output/report already exists; choose new paths')
    import torch
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError('Expected the MMDetection checkpoint state_dict')
    state, report = convert_state_dict(checkpoint['state_dict'])
    report['source'] = str(source)
    report['output'] = str(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=state, meta=dict(
        ocrdino_initialization='official_coco_r50_4scale_improved',
        source_checkpoint=source.name)), output)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
