"""Strict, case-sensitive Unicode character dictionary; blank is always 0."""
import json
import unicodedata
from pathlib import Path


DICTIONARY_VERSION = 'ocrdino.dictionary.v1'


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
    """Map Unicode code points to IDs 1..V (0 is reserved for CTC blank).

    ``dictionary`` is a JSON path or a sequence of one-code-point strings.
    JSON files contain ``schema_version`` and ``characters``. No casing,
    normalization, unknown substitution or whitespace stripping is performed.
    """

    blank_idx = 0

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
        self.num_classes = len(self.chars) + 1
        self._char_to_id = {char: i + 1 for i, char in enumerate(self.chars)}

    def encode(self, text):
        error = text_error(text)
        if error:
            raise ValueError(error)
        unknown = sorted(set(text).difference(self._char_to_id))
        if unknown:
            raise ValueError('Characters not present in dictionary: ' + repr(unknown))
        return [self._char_to_id[char] for char in text]

    def decode(self, indices):
        """Decode IDs, ignoring blank; callers perform CTC repeat collapse."""
        output = []
        for index in indices:
            value = int(index)
            if value != index or value < 0 or value >= self.num_classes:
                raise ValueError('Invalid character index: ' + repr(index))
            if value:
                output.append(self.chars[value - 1])
        return ''.join(output)

    def __len__(self):
        return len(self.chars)
