"""Whole-image text dataset with aggregate box and transcription annotations."""
import os.path as osp
import warnings
from collections import defaultdict

from mmengine.fileio import load

from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS

from .schema import validate_bbox, validate_document
from .vocabulary import CharacterDictionary, text_error


@DATASETS.register_module()
class OCRTextDataset(BaseDetDataset):
    """Read ``ocrdino.text.v1``; label 0 always means a complete text region.

    Invalid geometry is an error. The conversion tool moves such records to
    ``invalid_annotations`` and retains their images. Damaged text is kept for
    localization; valid text outside the dictionary raises an actionable error.
    """

    METAINFO = dict(classes=('text',), palette=[(0, 180, 255)])

    def __init__(self, *args, dictionary=None, **kwargs):
        if dictionary is None:
            raise ValueError('OCRTextDataset requires the same dictionary as the recognition head')
        self.dictionary = CharacterDictionary(dictionary)
        super().__init__(*args, **kwargs)

    def load_data_list(self):
        document = load(self.ann_file, backend_args=self.backend_args)
        images = validate_document(document)
        annotations = defaultdict(list)
        masked = 0
        for annotation in document['annotations']:
            image = images[annotation['image_id']]
            rec_ignore = annotation.get('rec_ignore', False) or annotation.get('ignore', False)
            if not rec_ignore:
                error = text_error(annotation['text'])
                if not error:
                    try:
                        self.dictionary.encode(annotation['text'])
                    except ValueError as exception:
                        raise ValueError('Annotation {} has valid text outside the configured '
                                         'dictionary: {}. Rebuild the dictionary from training '
                                         'JSON and use it for both dataset and recognition head.'
                                         .format(annotation['id'], exception)) from exception
                if error:
                    rec_ignore = True
                    masked += 1
            annotations[image['id']].append(dict(
                bbox=validate_bbox(annotation['bbox'], image['width'], image['height']),
                bbox_label=0,
                ignore_flag=int(annotation.get('ignore', False)),
                text=annotation['text'],
                rec_ignore=rec_ignore or annotation.get('ignore', False)))
        if masked:
            warnings.warn('{} annotations have damaged text: localization '
                          'retained, recognition disabled. Audit with prepare_dataset.py '
                          'and the training dictionary.'.format(masked), UserWarning)
        return [dict(
            img_id=image['id'],
            img_path=osp.join(self.data_prefix.get('img', ''), image['file_name']),
            height=image['height'], width=image['width'],
            annotation_complete=image.get('annotation_complete', True),
            instances=annotations[image['id']]) for image in document['images']]

    def filter_data(self):
        """Retain all images, including empty or partially annotated images."""
        return self.data_list
