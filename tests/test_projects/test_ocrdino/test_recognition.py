"""The copied project's retired CTC path must reject accidental imports."""
from pathlib import Path
import runpy
import pytest


def test_retired_recognition_path_requires_explicit_migration():
    root = Path(__file__).resolve().parents[3]
    with pytest.raises(ImportError, match='CharacterQueryHead'):
        runpy.run_path(str(root / 'projects/OCRDINO/ocrdino/recognition.py'))
