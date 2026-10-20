"""Archive safety, lazy nested ZIP handling, and lossless EXIF removal."""
import importlib.util
import io
import json
import stat
import tarfile
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / 'projects' / 'OCRDINO' / 'tools' / 'english_archives.py'
SPEC = importlib.util.spec_from_file_location('test_english_archive_helpers', MODULE_PATH)
archives = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archives)
try:
    from PIL import Image
except ImportError:
    Image = None


def nested_archive(root, name, entries):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, 'w') as archive:
        for filename, data in entries:
            archive.writestr(filename, data)
    with zipfile.ZipFile(root / (name + '.zip'), 'w') as archive:
        archive.writestr(name + '/train.json', json.dumps(dict(images=[])))
        archive.writestr(name + '/images.zip', inner.getvalue())


class TestEnglishArchives(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_annotation_only_never_unpacks_nested_zip(self):
        spool = self.root / 'spool'
        nested_archive(self.root, 'mlt2017', [('MLT_train_images/a.jpg', b'image')])
        with archives.DatasetSource(self.root, 'mlt2017', temp_dir=spool) as source:
            self.assertEqual(source.load_annotations('train'), dict(images=[]))
            self.assertFalse(spool.exists())
            destination = source.extract_image('a.jpg', 'train', self.root / 'out/a.jpg')
            self.assertEqual(destination.read_bytes(), b'image')
            paths = list(spool.iterdir())
            self.assertEqual(len(paths), 1)
            self.assertGreater(paths[0].stat().st_size, 0)
        self.assertEqual(list(spool.iterdir()), [])
        with self.assertRaises(RuntimeError):
            source.load_annotations('train')

    def test_syntext2_uses_outer_json_and_emcs_images(self):
        nested_archive(self.root, 'syntext2', [
            ('emcs_imgs/100.jpg', b'correct image'),
            ('annotations/ecms_v1_maxlen25.json', b'not authoritative')])
        with archives.DatasetSource(self.root, 'syntext2') as source:
            self.assertEqual(source.load_annotations('train'), dict(images=[]))
            output = source.extract_image('100.jpg', 'train', self.root / '100.jpg')
            self.assertEqual(output.read_bytes(), b'correct image')

    def test_totaltext_splits_are_separate_and_existing_files_not_overwritten(self):
        with zipfile.ZipFile(self.root / 'totaltext.zip', 'w') as archive:
            for split in ('train', 'test'):
                archive.writestr('totaltext/' + split + '.json', '{"images": []}')
                archive.writestr('totaltext/' + split + '_images/a.jpg', split.encode())
        with archives.DatasetSource(self.root, 'totaltext') as source:
            target = source.extract_image('a.jpg', 'train', self.root / 'a.jpg')
            self.assertEqual(target.read_bytes(), b'train')
            self.assertEqual(source.extract_image('a.jpg', 'train', target), target)
            with self.assertRaises(FileExistsError):
                source.extract_image('a.jpg', 'test', target)
            self.assertEqual(target.read_bytes(), b'train')
            self.assertEqual(list(self.root.glob('*.partial')), [])

    def test_tar_regular_file_and_symlink_handling(self):
        with tarfile.open(self.root / 'syntext1.tar', 'w') as archive:
            for filename, data in [
                    ('syntext1/train.json', b'{"images": []}'),
                    ('syntext1/images/syntext_word_eng/a.jpg', b'pixels')]:
                member = tarfile.TarInfo(filename)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            link = tarfile.TarInfo('syntext1/images/syntext_word_eng/link.jpg')
            link.type = tarfile.SYMTYPE
            link.linkname = '/etc/passwd'
            archive.addfile(link)
        with archives.DatasetSource(self.root, 'syntext1') as source:
            self.assertEqual(source.load_annotations('train'), dict(images=[]))
            target = source.extract_image('a.jpg', 'train', self.root / 'a.jpg')
            self.assertEqual(target.read_bytes(), b'pixels')
            with self.assertRaisesRegex(ValueError, 'regular file'):
                source.extract_image('link.jpg', 'train', self.root / 'bad.jpg')
            self.assertFalse((self.root / 'bad.jpg').exists())

    def test_traversal_and_unexpected_prefixes_rejected(self):
        nested_archive(self.root, 'mlt2017', [('a.jpg', b'one'), ('images/a.jpg', b'two')])
        with archives.DatasetSource(self.root, 'mlt2017') as source:
            for filename in ('../a.jpg', '/a.jpg', 'C:/a.jpg', 'a\\b.jpg', 'a/../b.jpg'):
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    source.extract_image(filename, 'train', self.root / 'output.jpg')
            self.assertIsNone(source._nested_path)
            with self.assertRaises(FileNotFoundError):
                source.extract_image('a.jpg', 'train', self.root / 'output.jpg')
            with self.assertRaises(ValueError):
                source.load_annotations('test')
        self.assertFalse((self.root / 'output.jpg').exists())

    def test_duplicate_entries_and_zip_symlinks_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            nested_archive(self.root, 'mlt2017', [
                ('MLT_train_images/a.jpg', b'one'), ('MLT_train_images/a.jpg', b'two')])
        with archives.DatasetSource(self.root, 'mlt2017') as source:
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                source.extract_image('a.jpg', 'train', self.root / 'a.jpg')
        info = zipfile.ZipInfo('totaltext/train_images/link.jpg')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(self.root / 'totaltext.zip', 'w') as archive:
            archive.writestr(info, '/etc/passwd')
        with archives.DatasetSource(self.root, 'totaltext') as source:
            with self.assertRaisesRegex(ValueError, 'regular file'):
                source.extract_image('link.jpg', 'train', self.root / 'bad.jpg')

    def test_invalid_nested_zip_is_cleaned_even_when_open_fails(self):
        spool = self.root / 'spool'
        with zipfile.ZipFile(self.root / 'mlt2017.zip', 'w') as archive:
            archive.writestr('mlt2017/images.zip', b'not a zip')
        with archives.DatasetSource(self.root, 'mlt2017', temp_dir=spool) as source:
            with self.assertRaises(zipfile.BadZipFile):
                source.extract_image('a.jpg', 'train', self.root / 'a.jpg')
            self.assertEqual(list(spool.iterdir()), [])

    @unittest.skipIf(Image is None, 'Pillow required for pixel equivalence check')
    def test_exif_removal_preserves_encoded_scan_and_decoded_pixels(self):
        image = Image.new('RGB', (9, 4), color=(24, 80, 143))
        exif = image.getexif()
        exif[274] = 6
        encoded = io.BytesIO()
        image.save(encoded, format='JPEG', exif=exif)
        original = encoded.getvalue()
        stripped = archives.strip_jpeg_exif(original)
        with Image.open(io.BytesIO(original)) as before, Image.open(io.BytesIO(stripped)) as after:
            self.assertEqual(before.getexif()[274], 6)
            self.assertNotIn(274, after.getexif())
            self.assertEqual(before.size, after.size)
            self.assertEqual(before.tobytes(), after.tobytes())
        self.assertEqual(original[original.index(b'\xff\xda'):],
                         stripped[stripped.index(b'\xff\xda'):])
        xmp_payload = b'http://ns.adobe.com/xap/1.0/\x00example'
        xmp = b'\xff\xe1' + (len(xmp_payload) + 2).to_bytes(2, 'big') + xmp_payload
        self.assertIn(xmp, archives.strip_jpeg_exif(original[:2] + xmp + original[2:]))
        self.assertEqual(archives.strip_jpeg_exif(stripped), stripped)

    def test_malformed_jpeg_fails_without_creating_output(self):
        for data in (b'png', b'\xff\xd8\xff', b'\xff\xd8\xff\xe1\x00\xffshort'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                archives.strip_jpeg_exif(data)


if __name__ == '__main__':
    unittest.main()
