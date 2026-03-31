"""Tests for song identification."""
from __future__ import annotations

import numpy as np
import pytest

from lyrasense.identification.identifier import SongIdentifier
from lyrasense.song_bank.manager import SongBank
from lyrasense.song_bank.song import Song
from lyrasense.audio.features import FeatureExtractor
from tests.conftest import make_demo_lyrics, make_sine_audio


def _make_bank_with_audio() -> tuple[SongBank, dict]:
    """Create a bank with two songs built from distinct sine tones."""
    bank = SongBank()
    sr = 22050
    lyrics = make_demo_lyrics()
    song_a = bank.add_song_from_audio("Song A", "Artist", make_sine_audio(440.0, 5.0, sr), sr, lyrics)
    song_b = bank.add_song_from_audio("Song B", "Artist", make_sine_audio(880.0, 5.0, sr), sr, lyrics)
    return bank, {"a": song_a, "b": song_b, "sr": sr}


class TestSongIdentifier:
    def test_identifies_correct_song(self) -> None:
        bank, songs = _make_bank_with_audio()
        identifier = SongIdentifier(bank, probe_seconds=2.0)
        # Probe with song A's audio – should identify Song A
        probe = make_sine_audio(440.0, 2.0, songs["sr"])
        result, score = identifier.identify(probe, songs["sr"])
        assert result is not None
        assert result.title == "Song A"
        assert 0.0 <= score <= 1.0

    def test_returns_none_for_empty_bank(self) -> None:
        bank = SongBank()
        identifier = SongIdentifier(bank, probe_seconds=1.0)
        probe = make_sine_audio(440.0, 1.0)
        result, score = identifier.identify(probe, 22050)
        assert result is None
        assert score == 0.0

    def test_all_scores_sorted(self) -> None:
        bank, songs = _make_bank_with_audio()
        identifier = SongIdentifier(bank, probe_seconds=1.0)
        probe = make_sine_audio(440.0, 1.0, songs["sr"])
        scores = identifier.identify_all_scores(probe, songs["sr"])
        assert len(scores) == 2
        # Scores are descending
        assert scores[0][1] >= scores[1][1]

    def test_short_probe_still_identifies(self) -> None:
        """Even a probe shorter than probe_seconds should not raise."""
        bank, songs = _make_bank_with_audio()
        identifier = SongIdentifier(bank, probe_seconds=5.0)
        # Only 0.5s audio
        probe = make_sine_audio(440.0, 0.5, songs["sr"])
        result, score = identifier.identify(probe, songs["sr"])
        assert result is not None
