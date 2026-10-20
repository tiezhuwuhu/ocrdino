"""Regression tests for the actual four-archive annotation conventions."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[3] / 'projects/OCRDINO/tools'
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    'english_conversion', TOOLS / 'prepare_english_datasets.py')
CONVERT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERT)


def rec(text):
    return [ord(char) - 32 for char in text] + [96] * (25 - len(text))


def document(text='BOOK'):
    return dict(
        images=[dict(id=9, file_name='00000009.jpg', width=100, height=80)],
        annotations=[dict(id=7, image_id=9, category_id=1, iscrowd=0,
                          bbox=[10, 20, 90, 20], rec=rec(text))])


class EnglishConversionTest(unittest.TestCase):
    def test_source_repeats_are_preserved(self):
        text, reasons = CONVERT.decode_rec(rec('BOOKkeeper!!'))
        self.assertEqual(text, 'BOOKkeeper!!')
        self.assertEqual(reasons, [])

    def test_unknown_char_is_not_deleted_to_form_wrong_word(self):
        codes = rec('ABCD')
        codes[1] = 95
        text, reasons = CONVERT.decode_rec(codes)
        self.assertEqual(text, 'A\ufffdCD')
        self.assertIn('unrecoverable_code_95', reasons)
        source = document()
        source['annotations'][0]['rec'] = codes
        converted, _ = CONVERT.convert_document(source, 'mlt2017', 'train')
        annotation = converted['annotations'][0]
        self.assertTrue(annotation['rec_ignore'])
        self.assertFalse(annotation['ignore'])
        self.assertEqual(annotation['raw_rec'], codes)

    def test_full25_is_explicitly_ambiguous_not_silently_truncated(self):
        codes = rec('A' * 25)
        text, reasons = CONVERT.decode_rec(codes)
        self.assertEqual(text, 'A' * 25)
        self.assertIn('full_25_slots_possible_truncation', reasons)
        self.assertEqual(CONVERT.decode_rec(codes, 'keep')[1], [])

    def test_empty_malformed_and_nontrailing_padding_are_masked(self):
        for codes, reason in [
                ([96] * 25, 'empty_recognition'),
                ([33, 96, 34] + [96] * 22, 'nontrailing_padding'),
                ([97] + [96] * 24, 'invalid_rec_code'),
                (None, 'invalid_rec_length')]:
            with self.subTest(reason=reason):
                source = document()
                source['annotations'][0]['rec'] = codes
                result, report = CONVERT.convert_document(source, 'syntext1', 'train')
                self.assertIn(reason, result['annotations'][0]['rec_ignore_reasons'])
                self.assertEqual(report['summary']['output_annotations'], 1)

    def test_marker_literals_not_arbitrarily_ignored(self):
        for text in ('***', '###'):
            output, _ = CONVERT.convert_document(document(text), 'syntext1', 'train')
            self.assertEqual(output['annotations'][0]['text'], text)
            self.assertFalse(output['annotations'][0]['ignore'])
            self.assertFalse(output['annotations'][0]['rec_ignore'])

    def test_border_clip_retains_original_box_and_text(self):
        source = document()
        source['annotations'][0]['bbox'] = [10, 20, 91, 61]
        output, report = CONVERT.convert_document(source, 'syntext1', 'train')
        annotation = output['annotations'][0]
        self.assertEqual(annotation['bbox'], [10., 20., 90., 60.])
        self.assertEqual(annotation['source_bbox'], [10, 20, 91, 61])
        self.assertEqual(annotation['text'], 'BOOK')
        self.assertEqual(report['summary']['clipped_bboxes'], 1)

    def test_invalid_box_keeps_image_and_raw_record(self):
        source = document()
        source['annotations'][0]['bbox'] = [110, 20, 30, 20]
        output, report = CONVERT.convert_document(source, 'syntext1', 'train')
        self.assertEqual(len(output['images']), 1)
        self.assertFalse(output['images'][0]['annotation_complete'])
        self.assertEqual(len(output['annotations']), 0)
        self.assertEqual(output['invalid_annotations'][0]['raw']['id'], 7)

    def test_verified_exif_frame_repair_recovers_outside_region(self):
        source = document()
        source['images'][0].update(id=228, file_name='0000228.jpg', width=294, height=392)
        source['annotations'][0].update(image_id=228, bbox=[374, 271, 17, 16])
        output, report = CONVERT.convert_document(source, 'totaltext', 'test')
        image = output['images'][0]
        self.assertEqual((image['width'], image['height']), (392, 294))
        self.assertEqual(image['image_repair'], 'strip_exif_without_rotating_pixels')
        self.assertEqual(output['annotations'][0]['bbox'], [374., 271., 17., 16.])
        self.assertEqual(report['summary']['image_frame_repairs'], 1)
        source = document()
        source['images'][0].update(file_name='0000228.jpg')
        with self.assertRaisesRegex(ValueError, 'Known EXIF case changed'):
            CONVERT.convert_document(source, 'totaltext', 'test')

    def test_id_namespace_and_train_only_merge(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for name in CONVERT.TRAIN_SOURCES:
                output, _ = CONVERT.convert_document(document(), name, 'train')
                path = Path(temporary) / (name + '_train.json')
                CONVERT.write_compact(path, output)
                paths.append(path)
            output_path = Path(temporary) / 'pretrain.json'
            totals = CONVERT.merge_pretraining(paths, output_path)
            merged = json.loads(output_path.read_text(encoding='utf-8'))
            CONVERT.prepare.schema.validate_document(merged)
            self.assertEqual(totals['images'], 3)
            self.assertEqual([a['image_id'] for a in merged['annotations']], [1, 2, 3])
            self.assertEqual([a['id'] for a in merged['annotations']], [1, 2, 3])
            self.assertEqual(len({i['file_name'] for i in merged['images']}), 3)
            with self.assertRaises(ValueError):
                CONVERT.merge_pretraining(paths + [Path('totaltext_test.json')], output_path)

    def test_query_eos_capacity_error_is_reported_not_silently_masked(self):
        source = document('A' * 24)
        source['annotations'][0]['bbox'] = [10, 20, 10, 50]
        output, report = CONVERT.convert_document(
            source, 'syntext1', 'train', query_buckets=(8, 16))
        self.assertFalse(report['ok'])
        self.assertEqual(report['errors'][0]['kind'], 'character_query_capacity')
        self.assertEqual(report['errors'][0]['required_positions'], 25)
        self.assertEqual(report['errors'][0]['query_bucket'], 8)
        self.assertFalse(output['annotations'][0]['rec_ignore'])


if __name__ == '__main__':
    unittest.main()
