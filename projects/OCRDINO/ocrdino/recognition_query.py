# Copyright (c) OpenMMLab. All rights reserved.
"""Ordered character queries over shared-C2 regional visual tokens."""

import math
from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from mmdet.registry import MODELS

from .recognition_capacity import (DEFAULT_BUCKETS, required_output_positions,
                                   validate_buckets, validate_width_scale)
from .vocabulary import CharacterDictionary, RECOGNITION_VERSION


def reduce_mean(value: Tensor) -> Tensor:
    """DDP mean without importing the detection ops for tensor-only tests."""
    if not dist.is_available() or not dist.is_initialized():
        return value
    result = value.clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result / dist.get_world_size()


def sinusoidal_positions(length: int, embed_dims: int, device=None,
                         dtype=torch.float32) -> Tensor:
    """Standard sine/cosine positions, shape ``(1, length, embed_dims)``.

    Frequencies and angles are calculated in FP32 even under AMP. This avoids
    half-precision rounding of the positions of long text sequences.
    """
    if length < 1 or embed_dims < 2 or embed_dims % 2:
        raise ValueError('Positions require positive length and even embed_dims.')
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, embed_dims, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.) / embed_dims))
    angles = position * frequencies[None, :]
    result = torch.empty((length, embed_dims), device=device, dtype=torch.float32)
    result[:, 0::2] = angles.sin()
    result[:, 1::2] = angles.cos()
    return result.unsqueeze(0).to(dtype=dtype)


class CharacterDecoderLayer(nn.Module):
    """Non-causal parallel decoder; positional visual keys, visual values."""

    def __init__(self, embed_dims, num_heads, ffn_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dims, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dims)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, queries: Tensor, visual: Tensor,
                visual_positions: Tensor) -> Tensor:
        # No causal mask and no attention-weight output. On supported PyTorch
        # backends this allows the memory-efficient scaled-dot-product path.
        attended = self.self_attn(
            queries, queries, queries, need_weights=False)[0]
        queries = self.norm1(queries + self.dropout1(attended))
        attended = self.cross_attn(
            queries, visual + visual_positions, visual,
            need_weights=False)[0]
        queries = self.norm2(queries + self.dropout2(attended))
        feedforward = self.linear2(self.dropout(F.relu(self.linear1(queries))))
        return self.norm3(queries + self.dropout3(feedforward))


@MODELS.register_module()
class CharacterQueryHead(nn.Module):
    """Four region convolutions and an ordered character-query decoder.

    RoIs are ``(batch_index, x1, y1, x2, y2)`` in resized input-image pixels.
    Visual sampling width and query count are independently selected from
    geometry, never from transcriptions. For every region, only characters
    and their following EOS are supervised; remaining slots have target -100.

    ``chunk_size`` bounds simultaneous regions in each pair of geometry
    buckets. In training ``with_cp=True`` recomputes each complete regional
    path, including its scalar loss, during backward. Merely looping over
    chunks without checkpointing would still retain all their activations.
    """

    ignore_index = -100
    required_output_positions = staticmethod(required_output_positions)

    def __init__(self,
                 dictionary: Union[str, Sequence[str]],
                 in_channels: int = 256,
                 hidden_channels: int = 128,
                 roi_height: int = 8,
                 width_buckets: Sequence[int] = DEFAULT_BUCKETS,
                 width_scale: float = 16.,
                 query_buckets: Sequence[int] = DEFAULT_BUCKETS,
                 query_width_scale: float = 16.,
                 embed_dims: int = 256,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 ffn_dim: int = 1024,
                 dropout: float = .1,
                 feature_stride: int = 4,
                 sampling_ratio: int = 2,
                 loss_weight: float = 1.,
                 lambda_eos: float = 1.,
                 gn_groups: int = 32,
                 chunk_size: int = 8,
                 with_cp: bool = True) -> None:
        super().__init__()
        self.width_buckets = validate_buckets(width_buckets, 'width_buckets')
        self.query_buckets = validate_buckets(query_buckets, 'query_buckets')
        self.width_scale = validate_width_scale(width_scale, 'width_scale')
        self.query_width_scale = validate_width_scale(
            query_width_scale, 'query_width_scale')
        if roi_height != 8:
            raise ValueError('CharacterQueryHead requires roi_height=8.')
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (in_channels, hidden_channels, embed_dims, num_layers,
                         num_heads, ffn_dim, gn_groups, chunk_size)):
            raise ValueError('Channels, layer counts and chunk_size must be positive integers.')
        if embed_dims % 2 or embed_dims % num_heads:
            raise ValueError('embed_dims must be even and divisible by num_heads.')
        if hidden_channels % gn_groups:
            raise ValueError('hidden_channels must be divisible by gn_groups.')
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('dropout must be finite and in [0, 1).')
        if not math.isfinite(feature_stride) or feature_stride <= 0:
            raise ValueError('feature_stride must be positive and finite.')
        if isinstance(sampling_ratio, bool) or not isinstance(sampling_ratio, int) or sampling_ratio < 0:
            raise ValueError('sampling_ratio must be a nonnegative integer.')
        if any(not math.isfinite(v) or v < 0 for v in (loss_weight, lambda_eos)):
            raise ValueError('loss_weight and lambda_eos must be finite and nonnegative.')
        self.dictionary = CharacterDictionary(dictionary)
        self.register_buffer('dictionary_hash', torch.tensor(
            list(self.dictionary.hash_digest), dtype=torch.uint8))
        self.register_buffer('recognition_version', torch.tensor(
            RECOGNITION_VERSION, dtype=torch.int64))
        self.in_channels = in_channels
        self.roi_height = roi_height
        self.embed_dims = embed_dims
        self.spatial_scale = 1. / feature_stride
        self.sampling_ratio = sampling_ratio
        self.loss_weight = float(loss_weight)
        self.lambda_eos = float(lambda_eos)
        self.chunk_size = chunk_size
        self.with_cp = bool(with_cp)
        self.projection = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1, bias=False)
        layers = []
        for stride in ((1, 1), (2, 1), (2, 1)):
            layers.extend([
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                          stride=stride, padding=1, bias=False),
                nn.GroupNorm(gn_groups, hidden_channels),
                nn.ReLU(inplace=True)])
        layers.extend([
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=(2, 1), bias=False),
            nn.ReLU(inplace=True)])
        self.convs = nn.Sequential(*layers)
        self.visual_projection = nn.Linear(hidden_channels, embed_dims)
        self.shared_content = nn.Parameter(torch.empty(1, 1, embed_dims))
        nn.init.normal_(self.shared_content, std=.02)
        self.decoder = nn.ModuleList([
            CharacterDecoderLayer(embed_dims, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)])
        self.classifier = nn.Linear(embed_dims, self.dictionary.num_classes)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version_key = prefix + 'recognition_version'
        hash_key = prefix + 'dictionary_hash'
        own_keys = [key[len(prefix):] for key in state_dict
                    if key.startswith(prefix)]
        components = ('projection.', 'convs.', 'lstm.', 'classifier.',
                      'visual_projection.', 'decoder.', 'shared_content')
        has_weights = any(key.startswith(components) for key in own_keys)
        if has_weights and version_key not in state_dict:
            raise RuntimeError(
                f'{prefix}recognition weights have no recognition_version. '
                'Old CTC recognition weights cannot initialize CharacterQueryHead; '
                'use a detection-only initialization or a complete query checkpoint.')
        if any(key.startswith('lstm.') for key in own_keys):
            raise RuntimeError(f'{prefix}old CTC/LSTM recognition weights are incompatible.')
        if version_key in state_dict and not torch.equal(
                state_dict[version_key].detach().cpu(),
                self.recognition_version.detach().cpu()):
            raise RuntimeError(f'{prefix}recognition_version is incompatible.')
        if has_weights and hash_key not in state_dict:
            raise RuntimeError(f'{prefix}recognition weights have no dictionary_hash.')
        if hash_key in state_dict and not torch.equal(
                state_dict[hash_key].detach().cpu(),
                self.dictionary_hash.detach().cpu()):
            # A direct exception is necessary: MMEngine may only log ordinary
            # load_state_dict error_msgs under strict=False.
            raise RuntimeError(f'{prefix}dictionary mismatch: checkpoint character order differs.')
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    @staticmethod
    def _geometry_counts(rois: Tensor, buckets: Tuple[int, ...],
                         width_scale: float) -> Tensor:
        if rois.ndim != 2 or rois.shape[1] != 5:
            raise ValueError('rois must have shape (M, 5).')
        if not len(rois):
            return rois.new_empty((0,), dtype=torch.long)
        geometry = rois.to(dtype=torch.float64)
        sizes = geometry[:, 3:5] - geometry[:, 1:3]
        if not torch.isfinite(geometry).all() or (sizes <= 0).any():
            raise ValueError('Every RoI must be finite with positive area.')
        desired = torch.ceil(sizes[:, 0] / sizes[:, 1] * width_scale)
        choices = geometry.new_tensor(buckets)
        indices = torch.bucketize(desired.contiguous(), choices, right=False)
        return choices[indices.clamp_max(len(buckets) - 1)].long()

    def roi_widths(self, rois: Tensor) -> Tensor:
        return self._geometry_counts(rois, self.width_buckets, self.width_scale)

    def roi_query_counts(self, rois: Tensor) -> Tensor:
        return self._geometry_counts(rois, self.query_buckets,
                                     self.query_width_scale)

    def _validate_inputs(self, c2: Tensor, rois: Tensor):
        if c2.ndim != 4 or c2.shape[1] != self.in_channels:
            raise ValueError(f'Expected C2 shape (B, {self.in_channels}, H, W).')
        if c2.device != rois.device:
            raise ValueError('C2 and RoIs must be on the same device.')
        widths, counts = self.roi_widths(rois), self.roi_query_counts(rois)
        if len(rois):
            ids = rois[:, 0]
            if ((ids != ids.floor()).any() or (ids < 0).any()
                    or (ids >= c2.shape[0]).any()):
                raise ValueError('RoI batch indices must index the C2 batch.')
        return widths, counts

    def validate_targets(self, rois: Tensor, texts: Sequence[str]) -> List[List[int]]:
        """Validate every effective GT before optional max_rois subsampling."""
        if len(texts) != len(rois):
            raise ValueError('One transcription is required for each RoI.')
        counts = self.roi_query_counts(rois).tolist()
        encoded = []
        for index, (text, count) in enumerate(zip(texts, counts)):
            ids = self.dictionary.encode(text)
            required = self.required_output_positions(text)
            if required > count:
                raise ValueError(
                    f'RoI {index}: text {text!r} needs {required} output positions '
                    f'(characters + EOS), but its geometric Query bucket is {count}. '
                    'Increase query_width_scale/query_buckets and verify the '
                    'visual resolution; text is never truncated.')
            encoded.append(ids)
        return encoded

    def encode_targets(self, texts: Sequence[str], num_queries: int, device=None):
        """Return ordered characters, one EOS, then loss-only ignore targets."""
        encoded = [self.dictionary.encode(text) for text in texts]
        return self._targets_from_encoded(encoded, num_queries, device)

    def _targets_from_encoded(self, encoded, num_queries, device):
        targets = torch.full((len(encoded), num_queries), self.ignore_index,
                             dtype=torch.long, device=device)
        lengths = torch.tensor([len(ids) for ids in encoded],
                               dtype=torch.long, device=device)
        for row, ids in enumerate(encoded):
            if not ids or len(ids) + 1 > num_queries:
                raise ValueError('A nonempty target and its EOS must fit the Query bucket.')
            targets[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            targets[row, len(ids)] = self.dictionary.eos_idx
        return targets, lengths

    def _region_losses(self, logits: Tensor, targets: Tensor,
                       lengths: Tensor) -> Tensor:
        """Character mean + weighted EOS for each region, without PAD loss."""
        with torch.autocast(device_type=logits.device.type, enabled=False):
            ce = F.cross_entropy(
                logits.float().transpose(1, 2), targets,
                reduction='none', ignore_index=self.ignore_index)
            character_mask = torch.arange(
                logits.shape[1], device=logits.device)[None, :] < lengths[:, None]
            character_loss = (ce * character_mask).sum(1) / lengths.float()
            eos_loss = ce.gather(1, lengths[:, None]).squeeze(1)
            return character_loss + self.lambda_eos * eos_loss

    def character_queries(self, batch_size: int, num_queries: int,
                          dtype=None) -> Tensor:
        """One shared learned content vector plus different ordinal positions."""
        content = self.shared_content
        if dtype is not None:
            content = content.to(dtype=dtype)
        positions = sinusoidal_positions(
            num_queries, self.embed_dims, content.device, content.dtype)
        return (content + positions).expand(batch_size, -1, -1)

    def decode_visual(self, visual: Tensor, num_queries: int) -> Tensor:
        """Attend to ordered visual tokens and output parallel character logits."""
        if visual.ndim != 3 or visual.shape[-1] != self.embed_dims:
            raise ValueError('visual must have shape (M, W, embed_dims).')
        queries = self.character_queries(len(visual), num_queries, visual.dtype)
        positions = sinusoidal_positions(
            visual.shape[1], self.embed_dims, visual.device, visual.dtype)
        for layer in self.decoder:
            queries = layer(queries, visual, positions)
        return self.classifier(queries)

    def _logits_from_projected(self, features: Tensor, rois: Tensor,
                               width: int, num_queries: int) -> Tensor:
        # Import only when regional sampling is actually used. Pure decoder,
        # target and loss tests do not require loading the compiled MMCV ops.
        from mmcv.ops import roi_align
        with torch.autocast(device_type=features.device.type, enabled=False):
            # MMCV 2.1 exports autograd.Function.apply, whose arguments must
            # be positional: input, rois, size, scale, samples, mode, aligned.
            regions = roi_align(
                features.float(), rois.float(), (self.roi_height, width),
                self.spatial_scale, self.sampling_ratio, 'avg', True)
        regional = self.convs(regions)
        if regional.shape[2] != 1:
            raise RuntimeError('The four-convolution head must collapse height to 1.')
        visual = self.visual_projection(regional.squeeze(2).transpose(1, 2))
        return self.decode_visual(visual, num_queries)

    def _groups(self, widths: Tensor, counts: Tensor):
        for width in self.width_buckets:
            for count in self.query_buckets:
                indices = ((widths == width) & (counts == count)).nonzero(
                    as_tuple=False).flatten()
                for start in range(0, len(indices), self.chunk_size):
                    yield indices[start:start + self.chunk_size], width, count

    def _iter_logits(self, c2: Tensor, rois: Tensor, widths: Tensor,
                     counts: Tensor):
        features = self.projection(c2).float()
        for indices, width, count in self._groups(widths, counts):
            yield indices, self._logits_from_projected(
                features, rois[indices], width, count)

    def _chunk_loss(self, features, rois, targets, lengths, width, count):
        logits = self._logits_from_projected(features, rois, width, count)
        return self._region_losses(logits, targets, lengths).sum()

    def _zero_loss(self, c2: Tensor) -> Tensor:
        zero = c2.reshape(-1)[:1].float().sum() * 0.
        for parameter in self.parameters():
            zero = zero + parameter.reshape(-1)[:1].float().sum() * 0.
        return zero

    def loss(self, c2: Tensor, rois: Tensor,
             texts: Sequence[str]) -> Dict[str, Tensor]:
        widths, counts = self._validate_inputs(c2, rois)
        encoded = self.validate_targets(rois, texts)
        # DDP averages gradients, so divide each rank's SUM by global_count /
        # world_size. A positive fractional mean must never be clamped to 1.
        normalizer = reduce_mean(c2.new_tensor(
            float(len(rois)), dtype=torch.float32)).clamp_min(1e-6)
        if not len(rois):
            return {'loss_rec': self.loss_weight * self._zero_loss(c2) / normalizer}
        features = self.projection(c2).float()
        loss_sum = features.new_zeros(())
        for indices, width, count in self._groups(widths, counts):
            selected = [encoded[index] for index in indices.tolist()]
            targets, lengths = self._targets_from_encoded(selected, count, c2.device)
            # Pass all chunk-dependent values as explicit arguments. A closure
            # over loop variables would recompute the wrong final chunk later.
            args = (features, rois[indices], targets, lengths, width, count)
            if self.with_cp and self.training and torch.is_grad_enabled():
                value = checkpoint(self._chunk_loss, *args,
                                   use_reentrant=False, preserve_rng_state=True)
            else:
                value = self._chunk_loss(*args)
            loss_sum = loss_sum + value
        return {'loss_rec': self.loss_weight * loss_sum / normalizer}

    @torch.no_grad()
    def predict(self, c2: Tensor, rois: Tensor) -> Tuple[List[str], Tensor, Tensor]:
        """Stop at first EOS; keep repeats and report missing termination.

        text_score is the geometric mean of emitted character probabilities,
        excluding EOS. Empty output has score zero; scores are not calibrated
        probabilities of whole-string correctness.
        """
        widths, counts = self._validate_inputs(c2, rois)
        texts = [''] * len(rois)
        scores = c2.new_zeros((len(rois),), dtype=torch.float32)
        terminated = torch.zeros((len(rois),), device=c2.device, dtype=torch.bool)
        if not len(rois):
            return texts, scores, terminated
        for indices, logits in self._iter_logits(c2, rois, widths, counts):
            log_probs, ids = logits.float().log_softmax(-1).max(-1)
            for row, original in enumerate(indices.tolist()):
                stops = (ids[row] == self.dictionary.eos_idx).nonzero(
                    as_tuple=False).flatten()
                end = int(stops[0]) if len(stops) else ids.shape[1]
                terminated[original] = bool(len(stops))
                texts[original] = self.dictionary.decode(ids[row, :end].tolist())
                if end:
                    scores[original] = log_probs[row, :end].mean().exp()
        return texts, scores, terminated
