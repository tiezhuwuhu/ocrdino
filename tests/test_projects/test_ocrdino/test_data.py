"""Run directly with Python; data-contract tests do not require torch/mmcv."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / 'projects' / 'OCRDINO'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vocabulary = load_module('test_ocr_vocabulary', PROJECT / 'ocrdino' / 'vocabulary.py')
schema = load_module('test_ocr_schema', PROJECT / 'ocrdino' / 'schema.py')
prepare = load_module('test_ocr_prepare', PROJECT / 'tools' / 'prepare_dataset.py')
builder = load_module('test_ocr_builder', PROJECT / 'tools' / 'build_dictionary.py')
auditor = load_module('test_ocr_auditor', PROJECT / 'tools' / 'audit_dataset.py')


def source():
    return dict(images=[dict(id=1, file_name='image.jpg', width=100, height=60),
                        dict(id=2, file_name='empty.jpg', width=80, height=40)],
                annotations=[dict(id=10, image_id=1, bbox=[10, 5, 40, 20], text='HELLO')])


class TestCharacterDictionary(unittest.TestCase):
    def test_case_sensitive_and_blank_separate_from_space(self):
        dictionary = vocabulary.CharacterDictionary([' ', 'A', 'a', '中'])
        self.assertEqual(dictionary.blank_idx, 0)
        self.assertEqual(dictionary.num_classes, 5)
        self.assertEqual(dictionary.encode('Aa 中'), [2, 3, 1, 4])
        self.assertEqual(dictionary.decode([0, 2, 2, 1, 4]), 'AA 中')

    def test_unknown_is_an_error_not_silent_replacement(self):
        dictionary = vocabulary.CharacterDictionary(['A'])
        with self.assertRaisesRegex(ValueError, 'not present'):
            dictionary.encode('AB')
        for invalid in (['A', 'A'], ['AB'], ['\ufffd'], ['\n'], []):
            with self.assertRaises(ValueError):
                vocabulary.CharacterDictionary(invalid)

    def test_shipped_english_has_all_95_printable_ascii_characters(self):
        dictionary = vocabulary.CharacterDictionary(PROJECT / 'dictionaries' / 'english.json')
        self.assertEqual(dictionary.chars, tuple(chr(i) for i in range(32, 127)))
        self.assertEqual(dictionary.num_classes, 96)


class TestConversion(unittest.TestCase):
    def test_invalid_geometry_keeps_image_and_original_record(self):
        raw = source()
        invalid = dict(id=11, image_id=1, bbox=[-1, -1, 0, 0], text='BAD')
        raw['annotations'].append(invalid)
        document, audit = prepare.convert_coco(raw)
        self.assertEqual(len(document['images']), 2)
        self.assertEqual(len(document['annotations']), 1)
        self.assertEqual(document['invalid_annotations'][0]['raw'], invalid)
        self.assertFalse(document['images'][0]['annotation_complete'])
        self.assertTrue(document['images'][1]['annotation_complete'])
        self.assertEqual(audit['summary']['incomplete_images'], 1)

    def test_bad_text_keeps_localization_and_unknown_dictionary_fails(self):
        raw = source()
        raw['annotations'][0]['text'] = '\ufffd\ufffd'
        document, _ = prepare.convert_coco(raw)
        item = document['annotations'][0]
        self.assertTrue(item['rec_ignore'])
        self.assertFalse(item['ignore'])
        self.assertTrue(document['images'][0]['annotation_complete'])
        dictionary = vocabulary.CharacterDictionary(['H', 'E', 'L'])
        with self.assertRaisesRegex(ValueError, 'annotation 10'):
            prepare.convert_coco(source(), dictionary=dictionary)

    def test_explicit_ignore_and_unreadable_placeholder(self):
        raw = source()
        raw['annotations'][0]['text'] = '###'
        document, _ = prepare.convert_coco(raw)
        self.assertTrue(document['annotations'][0]['ignore'])
        self.assertTrue(document['annotations'][0]['rec_ignore'])

    def test_out_of_bounds_quarantined_unless_explicitly_clipped(self):
        raw = source()
        raw['annotations'][0]['bbox'] = [-4, 5, 40, 20]
        document, _ = prepare.convert_coco(raw)
        self.assertEqual(len(document['annotations']), 0)
        document, audit = prepare.convert_coco(raw, clip_boxes=True)
        self.assertEqual(document['annotations'][0]['bbox'], [0, 5, 36, 20])
        self.assertEqual(document['annotations'][0]['raw_bbox'], [-4, 5, 40, 20])
        self.assertEqual(audit['summary']['events']['bbox_clipped'], 1)

    def test_infinite_geometry_is_quarantined_and_serialized_as_valid_json(self):
        raw = source()
        raw['annotations'][0]['bbox'][0] = float('nan')
        document, _ = prepare.convert_coco(raw)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'train.json'
            prepare.write_json(path, document)
            text = path.read_text(encoding='utf-8')
            self.assertIn('nonfinite_raw_value', text)
            json.loads(text, parse_constant=lambda value: self.fail('Invalid JSON constant: ' + value))

    def test_dictionary_sorted_and_ignores_unreliable_transcripts(self):
        document, _ = prepare.convert_coco(source())
        document['annotations'].append(dict(id=2, image_id=1, category_id=1,
                                            bbox=[5, 5, 10, 10], text='中', rec_ignore=True))
        dictionary, summary = builder.build_dictionary([document])
        self.assertEqual(dictionary['characters'], ['E', 'H', 'L', 'O'])
        self.assertEqual(summary['skipped_transcriptions'], 1)

    def test_polygon_envelope_and_degenerate_rejection(self):
        self.assertEqual(prepare.points_to_bbox([5, 1, 20, 4, 18, 10, 3, 7]), [3, 1, 17, 9])
        for points in ([-1] * 8, [0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 2, 0]):
            with self.assertRaises(ValueError):
                prepare.points_to_bbox(points)

    def test_rects_lines_only_and_images_without_json_retained(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow unavailable')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images, gt = root / 'images', root / 'gt'
            images.mkdir()
            gt.mkdir()
            Image.new('RGB', (100, 50)).save(images / 'one.jpg')
            Image.new('RGB', (100, 50)).save(images / 'two.jpg')
            prepare.write_json(gt / 'one.json', dict(
                chars=[dict(transcription='A', points=[1, 1, 2, 1, 2, 2, 1, 2])],
                lines=[dict(transcription='AB', points=[1, 1, 20, 1, 20, 10, 1, 10], ignore=0)]))
            document, audit = prepare.convert_rects(images, gt)
            self.assertEqual(len(document['annotations']), 1)
            self.assertEqual(document['annotations'][0]['text'], 'AB')
            self.assertEqual(len(document['images']), 2)
            self.assertFalse(document['images'][1]['annotation_complete'])
            self.assertEqual(audit['summary']['events']['missing_annotation_file'], 1)


class TestSchema(unittest.TestCase):
    def test_runtime_schema_rejects_invalid_geometry_and_duplicates(self):
        document, _ = prepare.convert_coco(source())
        for bbox in ([1, 1, 0, 2], [1, 1, 100, 2], [True, 1, 2, 3]):
            bad = copy.deepcopy(document)
            bad['annotations'][0]['bbox'] = bbox
            with self.assertRaises(ValueError):
                schema.validate_document(bad)
        bad = copy.deepcopy(document)
        bad['annotations'].append(copy.deepcopy(bad['annotations'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate annotation'):
            schema.validate_document(bad)

    def test_runtime_schema_rejects_path_escape(self):
        document, _ = prepare.convert_coco(source())
        for path in ('../outside.jpg', 'C:/outside.jpg', '/outside.jpg'):
            document['images'][0]['file_name'] = path
            with self.assertRaises(ValueError):
                schema.validate_document(document)

    def test_shipped_example_is_valid(self):
        document = prepare.read_json(PROJECT / 'examples' / 'train.example.json')
        schema.validate_document(document)
        dictionary = vocabulary.CharacterDictionary(PROJECT / 'dictionaries' / 'english.json')
        for item in document['annotations']:
            dictionary.encode(item['text'])


class TestAudit(unittest.TestCase):
    def setUp(self):
        self.dictionary = vocabulary.CharacterDictionary(PROJECT / 'dictionaries' / 'english.json')

    def test_adjacent_repeats_require_extra_ctc_steps(self):
        raw = source()
        raw['annotations'][0]['text'] = 'A' * 17
        document, _ = prepare.convert_coco(raw)
        report = auditor.audit_dataset(document, self.dictionary)
        self.assertFalse(report['ok'])
        self.assertEqual(report['errors'][0]['kind'], 'ctc_alignment')
        self.assertEqual(report['errors'][0]['required_steps'], 33)
        self.assertEqual(report['errors'][0]['input_steps'], 32)
        self.assertEqual(report['errors'][0]['annotation_id'], 1)

    def test_unknown_mixed_chinese_english_does_not_silently_train(self):
        raw = source()
        raw['annotations'][0]['text'] = 'A中B'
        document, _ = prepare.convert_coco(raw)
        report = auditor.audit_dataset(document, self.dictionary)
        self.assertFalse(report['ok'])
        self.assertEqual(report['errors'][0]['kind'], 'transcription')
        self.assertEqual(report['validrec'], 0)

    def test_audit_capacity_and_summary(self):
        document, _ = prepare.convert_coco(source())
        extra = copy.deepcopy(document['annotations'][0])
        extra['id'] = 2
        document['annotations'].append(extra)
        report = auditor.audit_dataset(document, self.dictionary, max_queries=1)
        self.assertEqual(report['maxroisperimage'], 2)
        self.assertEqual(report['widthbucketcounts'], {'32': 2})
        self.assertEqual(report['errors'][0]['kind'], 'query_capacity')

    def test_source_output_paths_cannot_collide(self):
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            prepare.guard_outputs(['train.json'], ['train.json'])
        with self.assertRaisesRegex(ValueError, 'different'):
            prepare.guard_outputs(['audit.json', 'audit.json'])
        with self.assertRaisesRegex(ValueError, 'outside'):
            prepare.guard_outputs(['images/one.jpg'], protected_roots=['images'])

    def test_damaged_surrogate_raw_text_serializes_without_data_loss(self):
        raw = source()
        raw['annotations'][0]['text'] = '\ud800'
        document, _ = prepare.convert_coco(raw)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'train.json'
            prepare.write_json(path, document)
            restored = prepare.read_json(path)
            self.assertEqual(restored['annotations'][0]['raw_text'], '\ud800')
            self.assertTrue(restored['annotations'][0]['rec_ignore'])


if __name__ == '__main__':
    unittest.main()
