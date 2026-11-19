"""A `%` in the video's name must not break the audio split.

ffmpeg's segment muxer reads its output name as a printf pattern, so a file
called "... 0% ..." made the split fail with "Invalid argument" and the video
could not be transcribed at all.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from modules.audio import transcript

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _tone(path, seconds):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={seconds}", str(path)], check=True)


@pytest.mark.parametrize("name", ["a 0% b.wav", "100%%.wav", "plain.wav"])
def test_split_survives_percent_in_the_name(tmp_path, name):
    src = tmp_path / name
    _tone(src, 3)
    chunks = transcript.split_audio(str(src), chunk_length=1)
    assert len(chunks) >= 3
    stem = name.rsplit(".", 1)[0]
    assert all(p.split("\\")[-1].split("/")[-1].startswith(stem + "_chunk_") for p in chunks)


def test_percent_in_the_folder_too(tmp_path):
    folder = tmp_path / "50% done"
    folder.mkdir()
    src = folder / "clip.wav"
    _tone(src, 2)
    assert len(transcript.split_audio(str(src), chunk_length=1)) >= 2
