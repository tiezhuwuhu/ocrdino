"""Build a deterministic character dictionary from valid training transcripts."""
import argparse
import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    '_ocrdino_prepare', Path(__file__).with_name('prepare_dataset.py'))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def build_dictionary(documents):
    characters = set()
    accepted, skipped = 0, 0
    for document in documents:
        prepare.schema.validate_document(document)
        for annotation in document['annotations']:
            text = annotation['text']
            if annotation.get('ignore', False) or annotation.get('rec_ignore', False) or prepare.vocabulary.text_error(text):
                skipped += 1
                continue
            characters.update(text)
            accepted += 1
    if not characters:
        raise ValueError('No valid recognition transcriptions found; cannot build dictionary')
    ordered = sorted(characters, key=ord)
    prepare.vocabulary.CharacterDictionary(ordered)
    return dict(schema_version=prepare.vocabulary.DICTIONARY_VERSION, characters=ordered), dict(
        characters=len(ordered), classes_with_eos=len(ordered) + 1, eos_id=len(ordered),
        accepted_transcriptions=accepted, skipped_transcriptions=skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True, help='Aggregated training JSON file(s)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare.guard_outputs([args.output], args.input)
    dictionary, summary = build_dictionary([prepare.read_json(path) for path in args.input])
    prepare.write_json(args.output, dictionary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
