"""Real CUDA forward/backward/optimizer/predict checks on the user's dataset."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--output', default='work_dirs/ocrdino_smoke.json')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    import torch
    from mmengine import Config
    from mmengine.runner import Runner, load_checkpoint
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    register_all_modules()
    cfg = Config.fromfile(args.config)
    checkpoint = args.checkpoint or cfg.load_from
    if not checkpoint:
        raise ValueError('Specify the converted initialization checkpoint')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This integration smoke requires CUDA and real MMCV CUDA ops')
    torch.cuda.set_device(device)
    model = MODELS.build(cfg.model)
    model.init_weights()
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.to(device).train()
    loader_cfg = cfg.train_dataloader.copy()
    loader_cfg.update(num_workers=0, persistent_workers=False, pin_memory=False)
    loader = Runner.build_dataloader(loader_cfg, seed=3407)
    if tuple(loader.dataset.dictionary.chars) != tuple(model.recognition_head.dictionary.chars):
        raise ValueError('Dataset/model dictionary mismatch')
    use_amp = cfg.optim_wrapper.type == 'AmpOptimWrapper'
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim_wrapper.optimizer.lr, weight_decay=.0001)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    torch.cuda.reset_peak_memory_stats(device)
    reports = []
    stream = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(stream)
        except StopIteration:
            stream = iter(loader)
            batch = next(stream)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            processed = model.data_preprocessor(batch, training=True)
            losses = model(**processed, mode='loss')
            total, log_vars = model.parse_losses(losses)
        if not torch.isfinite(total):
            raise RuntimeError('Non-finite training loss')
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient_norms = {}
        for name, module in [('backbone', model.backbone), ('encoder', model.encoder),
                             ('decoder', model.decoder), ('recognition', model.recognition_head)]:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise RuntimeError('Missing or non-finite gradients in ' + name)
            gradient_norms[name] = float(torch.sqrt(sum(g.float().square().sum() for g in grads)))
        torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        reports.append(dict(
            step=step, losses={key: float(value) for key, value in log_vars.items()},
            gradient_norms=gradient_norms,
            input_shape=list(processed['inputs'].shape),
            text_gt_counts=[len(s.gt_instances) for s in processed['data_samples']],
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024 ** 3,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024 ** 3))
    if not any(item['gradient_norms']['recognition'] > 0 for item in reports):
        raise RuntimeError('No recognition gradient: smoke batches need valid transcriptions')
    model.eval()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        predicted = model(**processed, mode='predict')
    report = dict(
        status='passed', device=torch.cuda.get_device_name(device), torch=torch.__version__,
        cuda=torch.version.cuda, amp=use_amp, checkpoint=str(checkpoint),
        steps=reports, prediction_counts=[len(s.pred_instances) for s in predicted],
        notes='Real training data and CUDA operators; no accuracy or convergence claim.')
    output = Path(args.output)
    if output.resolve() in [Path(args.config).resolve(), Path(checkpoint).resolve()]:
        raise ValueError('Report cannot overwrite config/checkpoint')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
