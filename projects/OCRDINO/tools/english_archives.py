"""Read the four supplied English archives without unpacking annotation audits.

Nested image ZIPs are copied to a temporary *disk file* only on the first image
request. No archive-wide bytes buffer or extractall() is used. The caller owns
the output-directory policy and image dimension/orientation checks.
"""
import hashlib
import io
import json
import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


_DATASETS = {
    'mlt2017': ('mlt2017.zip', ('train',)),
    'syntext1': ('syntext1.tar', ('train',)),
    'syntext2': ('syntext2.zip', ('train',)),
    'totaltext': ('totaltext.zip', ('train', 'test')),
}
_COPY_BUFFER = 1024 * 1024


def _relative_name(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Expected a nonempty POSIX relative image path')
    path = PurePosixPath(value)
    if (path.is_absolute() or any(part in ('', '.', '..') for part in value.split('/'))
            or any(':' in part for part in path.parts)
            or any(ord(char) < 32 for char in value)):
        raise ValueError('Unsafe archive path: ' + repr(value))
    return value


def strip_jpeg_exif(data):
    """Remove Exif APP1 segments, preserving all JPEG scan/pixel bytes exactly.

    This deliberately does not rotate/re-encode pixels. Non-Exif APP1 metadata,
    such as XMP, is retained. Segment lengths are checked before slicing.
    """
    if not data.startswith(b'\xff\xd8'):
        raise ValueError('EXIF removal requires a JPEG SOI marker')
    pieces = [data[:2]]
    position = 2
    while position < len(data):
        start = position
        if data[position] != 0xff:
            raise ValueError('Malformed JPEG marker before scan data')
        while position < len(data) and data[position] == 0xff:
            position += 1
        if position >= len(data):
            raise ValueError('Truncated JPEG marker')
        marker = data[position]
        position += 1
        if marker in (0xda, 0xd9):  # SOS: remainder is opaque entropy data.
            pieces.append(data[start:])
            return b''.join(pieces)
        if marker == 0x00:
            raise ValueError('Unexpected stuffed JPEG byte before scan data')
        if marker == 0x01 or 0xd0 <= marker <= 0xd8:
            pieces.append(data[start:position])
            continue
        if position + 2 > len(data):
            raise ValueError('Truncated JPEG segment length')
        length = int.from_bytes(data[position:position + 2], 'big')
        end = position + length
        if length < 2 or end > len(data):
            raise ValueError('Invalid JPEG segment length')
        payload = data[position + 2:end]
        if not (marker == 0xe1 and payload.startswith(b'Exif\x00\x00')):
            pieces.append(data[start:end])
        position = end
    raise ValueError('JPEG is missing scan/end marker')


def _stream_digest(stream):
    hasher = hashlib.sha256()
    length = 0
    while True:
        chunk = stream.read(_COPY_BUFFER)
        if not chunk:
            return length, hasher.digest()
        length += len(chunk)
        hasher.update(chunk)


def _digest(path):
    with Path(path).open('rb') as stream:
        return _stream_digest(stream)


def _copy_unique(stream, destination, strip_exif):
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError('Refusing a symlink destination: ' + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if strip_exif:
        stream = io.BytesIO(strip_jpeg_exif(stream.read()))
    try:
        # Exclusive creation prevents replacing either an existing file or a
        # symlink, including one created after the initial check.
        output = destination.open('xb')
    except FileExistsError:
        if (destination.is_symlink() or not destination.is_file()
                or _digest(destination) != _stream_digest(stream)):
            raise FileExistsError(
                'Existing image differs; refusing overwrite: ' + str(destination))
        return destination
    try:
        with output:
            shutil.copyfileobj(stream, output, length=_COPY_BUFFER)
    except BaseException:
        # This file was created by this call; remove its incomplete payload.
        destination.unlink(missing_ok=True)
        raise
    return destination


class DatasetSource:
    """Context-managed access to supplied archives.

    ``load_annotations(split)`` returns the original JSON document. The
    converter, not this module, decides annotation semantics.
    ``extract_image(file_name, split, destination, strip_exif=False)`` returns
    the output Path. Existing identical files are reusable; conflicts fail.
    """

    def __init__(self, source_root, name, temp_dir=None):
        if name not in _DATASETS:
            raise ValueError('Unknown English dataset: ' + str(name))
        self.name = name
        self.path = Path(source_root) / _DATASETS[name][0]
        self.temp_dir = None if temp_dir is None else Path(temp_dir)
        self._outer = None
        self._images = None
        self._image_index = None
        self._nested_path = None

    def __enter__(self):
        if self._outer is not None:
            raise RuntimeError('DatasetSource is already open')
        self._outer = (tarfile.open(self.path, 'r:') if self.name == 'syntext1'
                       else zipfile.ZipFile(self.path, 'r'))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        try:
            if self._images is not None and self._images is not self._outer:
                self._images.close()
        finally:
            try:
                if self._outer is not None:
                    self._outer.close()
            finally:
                self._images = self._outer = self._image_index = None
                if self._nested_path is not None:
                    self._nested_path.unlink(missing_ok=True)
                    self._nested_path = None

    def _check_open(self, split):
        if self._outer is None:
            raise RuntimeError('Use DatasetSource inside a with block')
        if split not in _DATASETS[self.name][1]:
            raise ValueError('Unsupported {} split: {}'.format(self.name, split))

    @staticmethod
    def _open_member(archive, member):
        _relative_name(member.name if isinstance(member, tarfile.TarInfo)
                       else member.filename)
        if isinstance(archive, tarfile.TarFile):
            if not member.isfile():
                raise ValueError('Archive member is not a regular file: ' + member.name)
            return archive.extractfile(member)
        mode = member.external_attr >> 16
        if member.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
            raise ValueError('Archive member is not a regular file: ' + member.filename)
        return archive.open(member, 'r')

    @staticmethod
    def _exact_member(archive, filename):
        members = (archive.getmembers() if isinstance(archive, tarfile.TarFile)
                   else archive.infolist())
        found = [member for member in members
                 if (member.name if isinstance(member, tarfile.TarInfo)
                     else member.filename) == filename]
        if not found:
            raise FileNotFoundError('Archive member is missing: ' + filename)
        if len(found) != 1:
            raise ValueError('Duplicate archive member: ' + filename)
        return found[0]

    def load_annotations(self, split):
        self._check_open(split)
        filename = '{}/{}.json'.format(self.name, split)
        member = self._exact_member(self._outer, filename)
        with self._open_member(self._outer, member) as stream:
            with io.TextIOWrapper(stream, encoding='utf-8-sig') as text:
                document = json.load(text)
        if not isinstance(document, dict):
            raise ValueError('Expected a JSON object in ' + filename)
        return document

    def _prepare_images(self):
        if self._images is not None:
            return
        if self.name not in ('mlt2017', 'syntext2'):
            self._images = self._outer
        else:
            member = self._exact_member(self._outer, self.name + '/images.zip')
            if self.temp_dir is not None:
                self.temp_dir.mkdir(parents=True, exist_ok=True)
            try:
                with tempfile.NamedTemporaryFile(
                        prefix='ocrdino-' + self.name + '-', suffix='.zip',
                        dir=self.temp_dir, delete=False) as output:
                    self._nested_path = Path(output.name)
                    with self._open_member(self._outer, member) as stream:
                        shutil.copyfileobj(stream, output, length=_COPY_BUFFER)
                self._images = zipfile.ZipFile(self._nested_path, 'r')
            except BaseException:
                if self._nested_path is not None:
                    self._nested_path.unlink(missing_ok=True)
                    self._nested_path = None
                raise
        members = (self._images.getmembers()
                   if isinstance(self._images, tarfile.TarFile)
                   else self._images.infolist())
        self._image_index = {}
        for member in members:
            name = (member.name if isinstance(member, tarfile.TarInfo)
                    else member.filename)
            self._image_index.setdefault(name, []).append(member)

    def extract_image(self, file_name, split, destination, strip_exif=False):
        self._check_open(split)
        file_name = _relative_name(file_name)
        self._prepare_images()
        if self.name == 'totaltext':
            prefixes = (self.name + '/' + split + '_images/',)
        elif self.name == 'syntext1':
            prefixes = ('syntext1/images/syntext_word_eng/',)
        elif self.name == 'syntext2':
            prefixes = ('emcs_imgs/',)
        else:
            prefixes = ('MLT_train_images/',)
        candidates = [prefix + file_name for prefix in prefixes]
        found = []
        for candidate in candidates:
            found.extend(self._image_index.get(candidate, ()))
        if not found:
            raise FileNotFoundError('Image {} is absent from {}'.format(file_name, self.path))
        if len(found) != 1:
            raise ValueError('Ambiguous/duplicate image member: ' + file_name)
        with self._open_member(self._images, found[0]) as stream:
            return _copy_unique(stream, destination, strip_exif)
