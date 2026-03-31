"""Tests for song data structures and SongBank."""
from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np
import pytest

from lyrasense.song_bank.song import LyricLine, Song
from lyrasense.song_bank.manager import SongBank
from tests.conftest import make_demo_lyrics, make_sine_audio


class TestLyricLine:
    def test_duration(self) -> None:
        line = LyricLine("Hello", start_time=1.0, end_time=3.5)
        assert line.duration == pytest.approx(2.5)

    def test_duration_missing(self) -> None:
        line = LyricLine("Hello")
        assert line.duration is None


class TestSong:
    def test_get_lyric_at_time_found(self) -> None:
        lyrics = make_demo_lyrics()
        song = Song("T", "A", lyrics=lyrics)
        assert song.get_lyric_at_time(1.0) == lyrics[0]
        assert song.get_lyric_at_time(3.9) == lyrics[1]

    def test_get_lyric_at_time_after_last(self) -> None:
        lyrics = make_demo_lyrics()
        song = Song("T", "A", lyrics=lyrics)
        # Beyond the last line's end_time → should return the last line whose start ≤ t
        result = song.get_lyric_at_time(20.0)
        assert result is not None

    def test_get_lyric_at_time_before_first(self) -> None:
        lyrics = make_demo_lyrics()
        song = Song("T", "A", lyrics=lyrics)
        assert song.get_lyric_at_time(-1.0) is None

    def test_time_to_frame_round_trip(self) -> None:
        song = Song("T", "A", sample_rate=22050, feature_hop_length=512)
        frame = song.time_to_frame(1.0)
        recovered = song.frame_to_time(frame)
        assert abs(recovered - 1.0) < 0.03  # within one frame

    def test_duration_from_lyrics(self) -> None:
        lyrics = make_demo_lyrics()
        song = Song("T", "A", lyrics=lyrics)
        assert song.duration == pytest.approx(12.0)

    def test_duration_no_timing(self) -> None:
        song = Song("T", "A", lyrics=[LyricLine("No timing")])
        assert song.duration is None


class TestSongBank:
    def test_add_and_get(self) -> None:
        bank = SongBank()
        song = Song("My Song", "Artist X")
        bank.add_song(song)
        assert bank.get("My Song", "Artist X") is song
        assert len(bank) == 1

    def test_add_song_from_audio(self) -> None:
        bank = SongBank()
        audio = make_sine_audio(duration=5.0)
        lyrics = make_demo_lyrics()
        song = bank.add_song_from_audio("Sine Song", "Test", audio, 22050, lyrics)
        assert song.reference_features is not None
        assert song.reference_features.ndim == 2

    def test_add_song_from_file(self) -> None:
        bank = SongBank()
        data = {
            "title": "File Song",
            "artist": "File Artist",
            "sample_rate": 22050,
            "lyrics": [
                {"text": "Line one", "start_time": 0.0, "end_time": 2.0},
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(data, fh)
            fname = fh.name

        song = bank.add_song_from_file(fname)
        assert song.title == "File Song"
        assert len(song.lyrics) == 1
        assert len(bank) == 1

    def test_add_song_from_file_inline_features(self) -> None:
        bank = SongBank()
        # Inline 3-frame × 12-feature matrix
        inline_feats = np.random.rand(3, 12).tolist()
        data = {
            "title": "Inline Song",
            "artist": "X",
            "lyrics": [],
            "reference_features": inline_feats,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(data, fh)
            fname = fh.name

        song = bank.add_song_from_file(fname)
        assert song.reference_features is not None
        assert song.reference_features.shape == (3, 12)

    def test_all_songs(self) -> None:
        bank = SongBank()
        bank.add_song(Song("A", "X"))
        bank.add_song(Song("B", "Y"))
        assert len(bank.all_songs()) == 2

    def test_case_insensitive_lookup(self) -> None:
        bank = SongBank()
        song = Song("My Song", "Artist X")
        bank.add_song(song)
        assert bank.get("MY SONG", "ARTIST X") is song
