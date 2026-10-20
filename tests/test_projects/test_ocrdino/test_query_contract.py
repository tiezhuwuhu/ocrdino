"""Dependency-free tests for query capacity, EOS IDs and checkpoint semantics."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

PROJECT = Path(__file__).resolve().parents[3] / 'projects/OCRDINO'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VOCAB = load('query_vocab_contract', PROJECT / 'ocrdino/vocabulary.py')
CAPACITY = load('query_capacity_contract', PROJECT / 'ocrdino/recognition_capacity.py')
AUDIT = load('query_audit_contract', PROJECT / 'tools/audit_dataset.py')


def document(text, bbox=(10, 10, 40, 20)):
    return dict(
        schema_version='ocrdino.text.v1', categories=[dict(id=1, name='text')],
        images=[dict(id=1, file_name='one.jpg', width=1000, height=1000)],
        annotations=[dict(id=1, image_id=1, category_id=1, bbox=list(bbox),
                          text=text, ignore=False, rec_ignore=False)])


class QueryCapacityTest(unittest.TestCase):
    def test_exact_bucket_boundaries_and_saturation(self):
        for requested, expected in [
                (20, 32), (32, 32), (32.01, 64), (50, 64), (110, 128),
                (200, 256), (350, 500), (500, 500), (501, 500)]:
            with self.subTest(requested=requested):
                self.assertEqual(
                    CAPACITY.select_geometry_bucket(requested, 16), expected)

    def test_capacity_reserves_eos_and_repeats_need_no_extra_slots(self):
        self.assertEqual(CAPACITY.required_output_positions('HELLO'), 6)
        self.assertEqual(CAPACITY.required_output_positions('LL'), 3)
        dictionary = VOCAB.CharacterDictionary(['A'])
        self.assertTrue(AUDIT.audit_dataset(document('A' * 31), dictionary)['ok'])
        report = AUDIT.audit_dataset(document('A' * 32), dictionary)
        self.assertFalse(report['ok'])
        self.assertEqual(report['errors'][0]['required_positions'], 33)
        self.assertEqual(report['errors'][0]['query_bucket'], 32)

    def test_499_chars_fit_max_bucket_but_500_need_explicitly_larger_capacity(self):
        dictionary = VOCAB.CharacterDictionary(['A'])
        self.assertTrue(AUDIT.audit_dataset(
            document('A' * 499, bbox=(0, 0, 1000, 20)), dictionary)['ok'])
        report = AUDIT.audit_dataset(
            document('A' * 500, bbox=(0, 0, 1000, 20)), dictionary)
        self.assertFalse(report['ok'])
        self.assertEqual(report['errors'][0]['required_positions'], 501)
        # Labels never enlarge the bucket; only explicit geometry configuration does.
        report = AUDIT.audit_dataset(
            document('A' * 500, bbox=(0, 0, 1000, 20)), dictionary,
            query_buckets=(32, 64, 128, 256, 512),
            visual_width_buckets=(32, 64, 128, 256, 512))
        self.assertTrue(report['ok'])

    def test_audit_checks_all_unmasked_gt_and_separate_visual_capacity(self):
        raw = document('A' * 33)
        raw['annotations'] += [dict(raw['annotations'][0], id=2, text='A')]
        dictionary = VOCAB.CharacterDictionary(['A'])
        report = AUDIT.audit_dataset(raw, dictionary)
        self.assertFalse(report['ok'])
        self.assertEqual(report['totalGT'], 2)
        raw['annotations'][0]['rec_ignore'] = True
        report = AUDIT.audit_dataset(raw, dictionary)
        self.assertTrue(report['ok'])
        self.assertEqual(report['recognition_ignored'], 1)
        report = AUDIT.audit_dataset(
            document('A' * 20), dictionary, visual_width_buckets=(8, 16))
        self.assertTrue(report['ok'])
        self.assertEqual(report['querybucketcounts'], {'32': 1})
        self.assertEqual(report['visualwidthbucketcounts'], {'16': 1})
        self.assertEqual(report['warnings'][0]['kind'], 'visual_resolution')

    def test_invalid_geometry_and_configuration_fail_explicitly(self):
        for buckets in [(), (64, 32), (32, 32), (True, 64), (0, 32)]:
            with self.subTest(buckets=buckets), self.assertRaises(ValueError):
                CAPACITY.validate_buckets(buckets)
        for scale in [0, -1, float('nan'), float('inf'), True]:
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                CAPACITY.validate_width_scale(scale)
        for width, height in [(0, 1), (1, 0), (float('nan'), 1), (True, 1)]:
            with self.assertRaises(ValueError):
                CAPACITY.select_geometry_bucket(width, height)


class QueryDictionaryContractTest(unittest.TestCase):
    def test_hello_target_and_first_eos_preserve_repeated_letters(self):
        dictionary = VOCAB.CharacterDictionary(['H', 'E', 'L', 'O'])
        self.assertEqual(dictionary.encode('HELLO') + [dictionary.eos_idx],
                         [0, 1, 2, 2, 3, 4])
        self.assertEqual(dictionary.decode([0, 1, 2, 2, 3, 4, 2]), 'HELLO')
        self.assertEqual(dictionary.decode([4, 2, 2]), '')
        self.assertEqual(dictionary.decode([2, 2]), 'LL')
        self.assertEqual(dictionary.num_classes, 5)
        self.assertFalse(hasattr(dictionary, 'blank_idx'))
        self.assertFalse(hasattr(dictionary, 'pad_idx'))

    def test_hash_preserves_exact_old_algorithm_and_character_order(self):
        chars = [' ', 'A', 'a', '中']
        dictionary = VOCAB.CharacterDictionary(chars)
        digest = hashlib.sha256(json.dumps(
            chars, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).digest()
        self.assertEqual(dictionary.hash_digest, digest)
        self.assertEqual(dictionary.hash_hex, digest.hex())
        self.assertNotEqual(dictionary.hash_digest,
                            VOCAB.CharacterDictionary(chars[::-1]).hash_digest)

    def test_metadata_accepts_detection_only_and_matching_query_checkpoint(self):
        dictionary = VOCAB.CharacterDictionary(['A', 'B'])
        VOCAB.validate_checkpoint_metadata(dict(state_dict={'backbone.weight': 1}), dictionary)
        checkpoint = dict(state_dict={'recognition_head.classifier.weight': 1},
                          meta=dict(ocrdino=VOCAB.recognition_metadata(dictionary)))
        VOCAB.validate_checkpoint_metadata(checkpoint, dictionary)
        self.assertNotIn('blank', checkpoint['meta']['ocrdino'])
        self.assertNotIn('pad', checkpoint['meta']['ocrdino'])

    def test_missing_old_or_wrong_dictionary_metadata_is_rejected(self):
        dictionary = VOCAB.CharacterDictionary(['A', 'B'])
        current = dict(state_dict={'module.recognition_head.classifier.weight': 1},
                       meta=dict(ocrdino=VOCAB.recognition_metadata(dictionary)))
        for bad in [
                dict(state_dict=current['state_dict']),
                dict(current, meta=dict(ocrdino=dict(
                    architecture='dino_c2_conv4_bilstm1_ctc', blank=0))),
                dict(current, meta=dict(ocrdino=VOCAB.recognition_metadata(
                    VOCAB.CharacterDictionary(['B', 'A']))))]:
            with self.assertRaises(RuntimeError):
                VOCAB.validate_checkpoint_metadata(bad, dictionary)
        bad = copy.deepcopy(current)
        bad['meta']['ocrdino']['recognition_version'] = 0
        with self.assertRaisesRegex(RuntimeError, 'recognition_version'):
            VOCAB.validate_checkpoint_metadata(bad, dictionary)


if __name__ == '__main__':
    unittest.main()