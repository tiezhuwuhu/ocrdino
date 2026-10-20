# Copyright (c) OpenMMLab. All rights reserved.
"""Small region-based recognition head sharing the detector's C2 features."""

import hashlib
import json
import math
from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.ops import roi_align
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.utils import reduce_mean

from .vocabulary import CharacterDictionary


@MODELS.register_module()
class LightCTCHead(nn.Module):
    """Four convolutions, one bidirectional LSTM and a CTC classifier.

    Boxes use input-image pixel coordinates, before any inverse resize, and
    have shape ``(M, 5)``: ``(batch_index, x1, y1, x2, y2)``. The shared C2
    feature is projected once, without upsampling the full feature map.

    Region widths depend only on box aspect ratio, both in training and at
    inference. Every width bucket is processed independently, so a reverse
    LSTM never sees right-padding from a different width. The last bucket
    caps the sampling width. Labels that cannot align to their chosen width
    raise an error instead of silently contributing zero recognition loss.

    Args:
        dictionary: CharacterDictionary JSON path or ordered character list.
        in_channels: Number of channels in the shared C2 feature.
        hidden_channels: Projection and convolution channels.
        lstm_hidden_size: Hidden size in each LSTM direction.
        roi_height: Sampling height; the four-convolution head requires 8.
        width_buckets: Increasing available sequence lengths.
        width_scale: Desired sequence length per unit box aspect ratio.
        feature_stride: Input-image pixels per shared-feature cell.
        sampling_ratio: Fixed RoIAlign samples per bin dimension.
        loss_weight: Multiplicative CTC loss weight.
        gn_groups: GroupNorm groups in the first three convolutions.
    """

    def __init__(self,
                 dictionary: Union[str, Sequence[str]],
                 in_channels: int = 256,
                 hidden_channels: int = 128,
                 lstm_hidden_size: int = 128,
                 roi_height: int = 8,
                 width_buckets: Sequence[int] = (32, 64, 128),
                 width_scale: float = 16.,
                 feature_stride: int = 4,
                 sampling_ratio: int = 2,
                 loss_weight: float = 1.,
                 gn_groups: int = 32) -> None:
        super().__init__()
        if roi_height != 8:
            raise ValueError('LightCTCHead requires roi_height=8.')
        if (not width_buckets or any(
                isinstance(w, bool) or not isinstance(w, int) or w < 1
                for w in width_buckets)
                or list(width_buckets) != sorted(set(width_buckets))):
            raise ValueError('width_buckets must be increasing positive ints.')
        if width_scale <= 0 or not math.isfinite(width_scale):
            raise ValueError('width_scale must be positive and finite.')
        if in_channels < 1 or lstm_hidden_size < 1 or hidden_channels < 1:
            raise ValueError('Channel counts and LSTM hidden size must be >0.')
        if gn_groups < 1 or hidden_channels % gn_groups:
            raise ValueError('hidden_channels must be divisible by gn_groups.')
        if feature_stride <= 0 or sampling_ratio < 0:
            raise ValueError('Invalid feature_stride or sampling_ratio.')
        if loss_weight < 0 or not math.isfinite(loss_weight):
            raise ValueError('loss_weight must be nonnegative and finite.')
        self.dictionary = CharacterDictionary(dictionary)
        digest = hashlib.sha256(
            json.dumps(
                list(self.dictionary.chars),
                ensure_ascii=False,
                separators=(',', ':')).encode('utf-8')).digest()
        self.register_buffer('dictionary_hash',
                             torch.tensor(list(digest), dtype=torch.uint8))
        self.in_channels = in_channels
        self.roi_height = roi_height
        self.width_buckets = tuple(width_buckets)
        self.width_scale = float(width_scale)
        self.spatial_scale = 1. / feature_stride
        self.sampling_ratio = sampling_ratio
        self.loss_weight = loss_weight
        self.projection = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1, bias=False)
        layers = []
        for stride in ((1, 1), (2, 1), (2, 1)):
            layers.extend([
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False),
                nn.GroupNorm(gn_groups, hidden_channels),
                nn.ReLU(inplace=True)
            ])
        layers.extend([
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(2, 1),
                bias=False),
            nn.ReLU(inplace=True)
        ])
        self.convs = nn.Sequential(*layers)
        self.lstm = nn.LSTM(
            hidden_channels,
            lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.classifier = nn.Linear(2 * lstm_hidden_size,
                                    self.dictionary.num_classes)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # A classifier can have the right shape but the wrong class ordering.
        # This must also reject strict=False loads, not just resumed training.
        hash_key = prefix + 'dictionary_hash'
        if hash_key in state_dict:
            saved_hash = state_dict[hash_key].detach().cpu()
            if not torch.equal(saved_hash, self.dictionary_hash.detach().cpu()):
                # MMEngine logs error_msgs instead of raising for strict=False.
                # A direct exception is required to protect classifier semantics.
                raise RuntimeError(
                    f'{prefix}dictionary mismatch: checkpoint character order '
                    'differs from the configured dictionary. Use the original '
                    'dictionary or explicitly migrate the recognition weights.')
        elif any(
                key.startswith(prefix + component + '.')
                for key in state_dict
                for component in ('projection', 'convs', 'lstm', 'classifier')):
            raise RuntimeError(
                f'{prefix}recognition weights have no dictionary_hash; their '
                'character ordering cannot be verified. Load a converted '
                'detection-only checkpoint or a complete OCRDINO checkpoint.')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @staticmethod
    def required_ctc_steps(text: str) -> int:
        """Minimum time steps, including blanks between adjacent repeats."""
        return len(text) + sum(a == b for a, b in zip(text, text[1:]))

    def roi_widths(self, rois: Tensor) -> Tensor:
        """Choose a width from geometry alone; also used during inference."""
        if rois.ndim != 2 or rois.shape[1] != 5:
            raise ValueError('rois must have shape (M, 5).')
        if rois.numel() == 0:
            return rois.new_empty((0, ), dtype=torch.long)
        sizes = rois[:, 3:5] - rois[:, 1:3]
        if not torch.isfinite(rois).all() or (sizes <= 0).any():
            raise ValueError('Every RoI must be finite with positive area.')
        desired = torch.ceil(sizes[:, 0] / sizes[:, 1] * self.width_scale)
        buckets = rois.new_tensor(self.width_buckets)
        indices = torch.bucketize(desired.contiguous(), buckets)
        return buckets[indices.clamp_max(len(self.width_buckets) - 1)].long()

    def _validate_inputs(self, c2: Tensor, rois: Tensor) -> Tensor:
        if c2.ndim != 4 or c2.shape[1] != self.in_channels:
            raise ValueError(
                f'Expected C2 shape (B, {self.in_channels}, H, W).')
        if c2.device != rois.device:
            raise ValueError('C2 and RoIs must be on the same device.')
        widths = self.roi_widths(rois)
        if len(rois):
            batch_ids = rois[:, 0]
            if ((batch_ids != batch_ids.floor()).any()
                    or (batch_ids < 0).any()
                    or (batch_ids >= c2.shape[0]).any()):
                raise ValueError('RoI batch indices must index the C2 batch.')
        return widths

    def _iter_logits(self, c2: Tensor, rois: Tensor, widths: Tensor):
        # Projection and regional convolutions may use the caller's AMP mode.
        # Cast once, not once per bucket: all buckets share this feature map.
        features = self.projection(c2).float()
        sampling_rois = rois.float()
        for width in self.width_buckets:
            indices = (widths == width).nonzero(as_tuple=False).flatten()
            if not len(indices):
                continue
            # FP32 also avoids RoI/feature dtype mismatches in MMCV kernels.
            with torch.autocast(device_type=c2.device.type, enabled=False):
                regions = roi_align(
                    features,
                    sampling_rois[indices],
                    output_size=(self.roi_height, width),
                    spatial_scale=self.spatial_scale,
                    sampling_ratio=self.sampling_ratio,
                    pool_mode='avg',
                    aligned=True)
            sequences = self.convs(regions).squeeze(2).transpose(1, 2)
            # LSTM and CTC remain FP32, including when the detector uses AMP.
            with torch.autocast(device_type=c2.device.type, enabled=False):
                sequences, _ = self.lstm(sequences.float().contiguous())
                logits = self.classifier(sequences)
            yield indices, logits

    def _zero_loss(self, c2: Tensor) -> Tensor:
        # Touch every parameter even on ranks with no recognizable regions.
        # DDP can then use the same graph participation without unused params.
        zero = c2.reshape(-1)[:1].float().sum() * 0.
        for parameter in self.parameters():
            zero = zero + parameter.reshape(-1)[:1].float().sum() * 0.
        return zero

    def loss(self, c2: Tensor, rois: Tensor,
             texts: Sequence[str]) -> Dict[str, Tensor]:
        """CTC mean per region, normalized consistently across DDP ranks.

        Recognition-ignored regions must be removed by the caller. Invalid,
        empty or unknown transcriptions raise; they are never silently erased.
        CTC loss is divided by target length, then averaged over all regions
        across ranks, matching PyTorch's ``CTCLoss(reduction='mean')``.
        """
        widths = self._validate_inputs(c2, rois)
        if len(texts) != len(rois):
            raise ValueError('One transcription is required for each RoI.')
        targets: List[List[int]] = []
        for index, (text, width) in enumerate(zip(texts, widths.tolist())):
            if not isinstance(text, str) or not text:
                raise ValueError(f'RoI {index} needs a nonempty string.')
            encoded = self.dictionary.encode(text)
            required = self.required_ctc_steps(text)
            if required > width:
                raise ValueError(
                    f'RoI {index}: {text[:60]!r} needs at least {required} '
                    f'CTC steps, but its geometric width bucket is {width}. '
                    'Increase width_scale and/or width_buckets; check that '
                    'the text box contains one correctly annotated line.')
            targets.append(encoded)
        # Every rank participates, including an empty local batch. Do not
        # clamp to 1: a positive global mean count can be below 1 in DDP.
        normalizer = reduce_mean(
            c2.new_tensor(float(len(rois)), dtype=torch.float32)).clamp_min(1e-6)
        if not len(rois):
            return {'loss_rec_ctc': self._zero_loss(c2) / normalizer}
        loss_sum = c2.new_zeros((), dtype=torch.float32)
        for indices, logits in self._iter_logits(c2, rois, widths):
            selected = [targets[i] for i in indices.tolist()]
            target_lengths = torch.tensor([len(t) for t in selected],
                                          dtype=torch.long)
            flat_targets = torch.tensor(
                [i for target in selected for i in target],
                dtype=torch.long,
                device=logits.device)
            input_lengths = torch.full((len(selected), ),
                                       logits.shape[1],
                                       dtype=torch.long)
            with torch.autocast(device_type=c2.device.type, enabled=False):
                log_probs = F.log_softmax(logits.float(), dim=-1)
                loss = F.ctc_loss(
                    log_probs.transpose(0, 1).contiguous(),
                    flat_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.dictionary.blank_idx,
                    reduction='none',
                    zero_infinity=False)
                loss_sum = loss_sum + (
                    loss / target_lengths.to(loss.device)).sum()
        return {'loss_rec_ctc': self.loss_weight * loss_sum / normalizer}

    @torch.no_grad()
    def predict(self, c2: Tensor,
                rois: Tensor) -> Tuple[List[str], Tensor]:
        """Greedy CTC decoding, preserving input box order.

        A score is the geometric mean probability at emitted character steps
        (zero for an empty decoding). It is separate from detection confidence
        and is not a calibrated probability that the whole word is correct.
        """
        widths = self._validate_inputs(c2, rois)
        texts = [''] * len(rois)
        scores = c2.new_zeros((len(rois), ), dtype=torch.float32)
        if not len(rois):
            return texts, scores
        for indices, logits in self._iter_logits(c2, rois, widths):
            max_log_probs, ids = logits.float().log_softmax(-1).max(-1)
            for row, original_index in enumerate(indices.tolist()):
                previous = torch.cat([ids.new_full((1, ), -1), ids[row, :-1]])
                emitted = ((ids[row] != previous)
                           & (ids[row] != self.dictionary.blank_idx))
                texts[original_index] = self.dictionary.decode(
                    ids[row, emitted].tolist())
                if emitted.any():
                    char_log_probs = max_log_probs[row, emitted]
                    scores[original_index] = char_log_probs.mean().exp()
        return texts, scores
