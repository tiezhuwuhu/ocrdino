"""Strict Unicode character IDs 0..V-1 and EOS=V; no blank/PAD classes."""
import hashlib
import json
import unicodedata
from pathlib import Path


DICTIONARY_VERSION = 'ocrdino.dictionary.v1'
QUERY_ARCHITECTURE = 'dino_c2_conv4_charquery3_ce'
RECOGNITION_VERSION = 1
LABEL_LAYOUT = 'characters_0_to_v_minus_1_eos_v'


def text_error(text):
    """Return a reason for unusable transcription, without changing its text."""
    if not isinstance(text, str):
        return 'text is not a string'
    if not text or not text.strip():
        return 'empty transcription'
    if '\ufffd' in text:
        return 'Unicode replacement character in transcription'
    if any(unicodedata.category(char) in ('Cc', 'Cs', 'Cn') for char in text):
        return 'control, surrogate or unassigned code point in transcription'
    return None


class CharacterDictionary:
    """Map ordered Unicode code points to IDs 0..V-1, reserving V for EOS.

    JSON files retain their existing dictionary.v1 character ordering. No
    casing, normalization, unknown substitution or whitespace stripping occurs.
    The file format is unchanged; the recognition checkpoint version explicitly
    distinguishes this class layout from old CTC weights.
    """

    def __init__(self, dictionary):
        if isinstance(dictionary, (str, Path)):
            with open(dictionary, encoding='utf-8-sig') as handle:
                dictionary = json.load(handle)
            if not isinstance(dictionary, dict) or dictionary.get(
                    'schema_version') != DICTIONARY_VERSION:
                raise ValueError('Dictionary JSON must use ' + DICTIONARY_VERSION)
            dictionary = dictionary.get('characters')
        if not isinstance(dictionary, (list, tuple)) or not dictionary:
            raise ValueError('characters must be a nonempty list of characters')
        for char in dictionary:
            if not isinstance(char, str) or len(char) != 1:
                raise ValueError('Each dictionary entry must be one Unicode code point')
            if char == '\ufffd' or unicodedata.category(char) in ('Cc', 'Cs', 'Cn'):
                raise ValueError('Invalid dictionary character: ' + repr(char))
        if len(set(dictionary)) != len(dictionary):
            raise ValueError('Duplicate dictionary characters are not allowed')
        self.chars = tuple(dictionary)
        self.eos_idx = len(self.chars)
        self.num_classes = len(self.chars) + 1
        self._char_to_id = {char: i for i, char in enumerate(self.chars)}
        self.hash_digest = hashlib.sha256(json.dumps(
            list(self.chars), ensure_ascii=False,
            separators=(',', ':')).encode('utf-8')).digest()
        self.hash_hex = self.hash_digest.hex()

    def encode(self, text):
        """Return only real-character IDs; the recognition head appends EOS."""
        error = text_error(text)
        if error:
            raise ValueError(error)
        unknown = sorted(set(text).difference(self._char_to_id))
        if unknown:
            raise ValueError('Characters not present in dictionary: ' + repr(unknown))
        return [self._char_to_id[char] for char in text]

    def decode(self, indices):
        """Read ordered IDs until the first EOS, preserving adjacent repeats."""
        output = []
        for index in indices:
            value = int(index)
            if isinstance(index, bool) or value != index or value < 0 or value >= self.num_classes:
                raise ValueError('Invalid character index: ' + repr(index))
            if value == self.eos_idx:
                break
            output.append(self.chars[value])
        return ''.join(output)

    def __len__(self):
        return len(self.chars)


def recognition_metadata(dictionary):
    """Serializable recognition semantics; saved alongside the state buffers."""
    return dict(architecture=QUERY_ARCHITECTURE,
                recognition_version=RECOGNITION_VERSION, label_layout=LABEL_LAYOUT,
                characters=list(dictionary.chars), eos_id=dictionary.eos_idx,
                num_classes=dictionary.num_classes, ignore_index=-100,
                dictionary_hash=dictionary.hash_hex, bbox_format='xyxy',
                dictionary_case_sensitive=True)


def validate_checkpoint_metadata(checkpoint, dictionary):
    """Reject incompatible OCR metadata before model tensors are loaded.

    Detection-only initialization needs no OCR metadata. Runner checkpoints
    containing recognition tensors must include the new metadata; direct
    state_dict loads are independently guarded by the head's version/hash.
    """
    if not isinstance(checkpoint, dict):
        raise RuntimeError('Expected a checkpoint dictionary')
    meta = checkpoint.get('meta', {})
    if not isinstance(meta, dict):
        raise RuntimeError('Checkpoint meta must be a dictionary')
    saved = meta.get('ocrdino')
    state = checkpoint.get('state_dict', checkpoint)
    has_recognition = isinstance(state, dict) and any(
        name.removeprefix('module.').startswith('recognition_head.')
        for name in state if isinstance(name, str))
    if saved is None:
        if has_recognition:
            raise RuntimeError('Recognition checkpoint has no OCRDINO metadata; '
                               'use a new Character Query checkpoint or converted detection-only weights')
        return
    if not isinstance(saved, dict):
        raise RuntimeError('OCRDINO checkpoint metadata must be a dictionary')
    expected = recognition_metadata(dictionary)
    if any(key in saved for key in ('blank', 'blank_idx', 'pad', 'pad_idx')):
        raise RuntimeError('Old blank/PAD recognition metadata is incompatible with Character Query EOS')
    for key, value in expected.items():
        if saved.get(key) != value:
            raise RuntimeError('OCRDINO checkpoint metadata mismatch for {}: expected {!r}, got {!r}'
                               .format(key, value, saved.get(key)))