"""Keep transcriptions aligned with detection instances during packing."""
import numpy as np
import torch

from mmdet.datasets.transforms import LoadAnnotations, PackDetInputs
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class OCRLoadAnnotations(LoadAnnotations):
    """Load boxes and their corresponding transcription/recognition flags.

    The supplied pipeline only resizes. Any future transform which filters or
    reorders boxes must apply the identical index to gt_texts/gt_rec_ignore.
    """

    def __init__(self, *args, box_type=None, **kwargs):
        if box_type is not None:
            raise ValueError('OCRDINO expects tensor boxes; use box_type=None')
        super().__init__(*args, box_type=None, **kwargs)

    def transform(self, results):
        results = super().transform(results)
        results['gt_texts'] = [instance['text'] for instance in results['instances']]
        results['gt_rec_ignore'] = np.asarray([
            instance.get('rec_ignore', False) for instance in results['instances']
        ], dtype=np.bool_)
        return results


@TRANSFORMS.register_module()
class OCRPackDetInputs(PackDetInputs):
    """Pack text fields using exactly the same ignore split as the boxes."""

    def __init__(self, meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                'scale_factor', 'flip', 'flip_direction',
                                'annotation_complete')):
        super().__init__(meta_keys=meta_keys)

    def transform(self, results):
        if 'gt_texts' in results:
            num_boxes = len(results['gt_bboxes'])
            if len(results['gt_texts']) != num_boxes or len(results['gt_rec_ignore']) != num_boxes:
                raise ValueError('OCR text fields and boxes became misaligned in the pipeline')
        packed = super().transform(results)
        if 'gt_texts' not in results:
            return packed
        flags = results.get('gt_ignore_flags', np.zeros(len(results['gt_texts']), dtype=np.bool_))
        data_sample = packed['data_samples']
        for ignored, instances in ((False, data_sample.gt_instances),
                                   (True, data_sample.ignored_instances)):
            indices = np.flatnonzero(np.asarray(flags, dtype=np.bool_) == ignored)
            instances.texts = [results['gt_texts'][int(index)] for index in indices]
            instances.rec_ignore = torch.as_tensor(results['gt_rec_ignore'][indices], dtype=torch.bool)
        return packed
