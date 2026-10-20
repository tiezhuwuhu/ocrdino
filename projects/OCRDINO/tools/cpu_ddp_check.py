"""Two-process CPU/Gloo check of unequal-region recognition normalization."""
import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def worker(rank, store_uri, output):
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel
    from projects.OCRDINO.ocrdino.recognition_query import CharacterQueryHead
    torch.set_num_threads(2)
    torch.manual_seed(53)

    def head():
        return CharacterQueryHead(
            dictionary=['A', 'B'], in_channels=4, hidden_channels=8,
            embed_dims=16, num_layers=2, num_heads=4, ffn_dim=32,
            gn_groups=4, dropout=0., width_buckets=(8, 16, 32),
            query_buckets=(8, 16, 32), width_scale=8., query_width_scale=8.,
            chunk_size=1, with_cp=True)

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head()
        def forward(self, c2, rois, texts):
            return self.head.loss(c2, rois, texts)['loss_rec']

    dist.init_process_group(
        'gloo', init_method=store_uri, rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=120))
    try:
        distributed = DistributedDataParallel(Wrapper(), find_unused_parameters=False)
        reference = head()
        reference.load_state_dict(distributed.module.head.state_dict())
        torch.manual_seed(900)
        features = torch.randn(2, 4, 16, 32)
        reports = []
        # Uneven valid counts, one completely empty rank, then both empty.
        for counts in ((1, 2), (0, 1), (0, 0)):
            global_rois, all_texts = [], []
            for image, count in enumerate(counts):
                for region in range(count):
                    global_rois.append([image, 4., 4., 36. + region * 32., 36.])
                    all_texts.append('AAB' if region else 'AB')
            all_rois = torch.tensor(global_rois, dtype=torch.float32).reshape(-1, 5)
            reference.zero_grad(set_to_none=True)
            reference_features = features.clone().requires_grad_()
            # Both ranks evaluate the same global batch, so reduce_mean(count)
            # equals global_count. This is the exact single-batch objective.
            expected_loss = reference.loss(reference_features, all_rois, all_texts)['loss_rec']
            expected_loss.backward()
            distributed.zero_grad(set_to_none=True)
            local_features = features[rank:rank + 1].clone().requires_grad_()
            selected = all_rois[:, 0] == rank
            local_rois = all_rois[selected].clone()
            local_rois[:, 0] = 0.
            local_texts = [text for text, use in zip(all_texts, selected.tolist()) if use]
            loss = distributed(local_features, local_rois, local_texts)
            loss.backward()
            average = loss.detach().clone()
            dist.all_reduce(average)
            average /= 2
            torch.testing.assert_close(average, expected_loss.detach(), rtol=2e-5, atol=2e-6)
            maximum_error = 0.
            for (name, actual), (other_name, expected) in zip(
                    distributed.module.head.named_parameters(), reference.named_parameters()):
                assert name == other_name
                assert actual.grad is not None and expected.grad is not None, name
                assert torch.isfinite(actual.grad).all(), name
                torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-4, atol=3e-6,
                                           msg=lambda msg: name + ': ' + msg)
                maximum_error = max(maximum_error, float((actual.grad - expected.grad).abs().max()))
            torch.testing.assert_close(
                local_features.grad / 2, reference_features.grad[rank:rank + 1],
                rtol=3e-4, atol=3e-6)
            reports.append(dict(
                regions_per_rank=list(counts), average_loss=float(average),
                global_reference_loss=float(expected_loss.detach()),
                maximum_parameter_gradient_error=maximum_error,
                all_parameters_participated=True))
        if rank == 0:
            report = dict(
                status='passed', backend='gloo', world_size=2, device='cpu',
                torch=torch.__version__, activation_checkpointing=True,
                chunk_size=1, dropout=0.,
                scope='Real DDP recognition head, including actual MMCV RoIAlign; not full-detector DDP or CUDA',
                cases=reports)
            Path(output).write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report, indent=2), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='projects/OCRDINO/reports/query_validation/cpu_ddp.json')
    args = parser.parse_args()
    import torch.multiprocessing as mp
    import uuid
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    store = ROOT / '.test_tmp' / ('gloo_' + uuid.uuid4().hex)
    store.parent.mkdir(parents=True, exist_ok=True)
    mp.spawn(worker, args=(store.as_uri(), str(output)), nprocs=2, join=True)


if __name__ == '__main__':
    main()

