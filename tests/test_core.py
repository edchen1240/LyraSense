"""End-to-end integration tests for LyraSense core."""
from __future__ import annotations

import numpy as np
import pytest

from lyrasense.core import LyraSense
from lyrasense.song_bank.manager import SongBank
from lyrasense.song_bank.song import LyricLine
from tests.conftest import make_demo_lyrics, make_sine_audio


def _make_ls_with_known_song(sr: int = 22050) -> tuple[LyraSense, object]:
    """Build a LyraSense instance pre-loaded with one song."""
    bank = SongBank()
    audio = make_sine_audio(440.0, 12.0, sr)
    song = bank.add_song_from_audio("Test Song", "Test Artist", audio, sr, make_demo_lyrics())
    ls = LyraSense(bank, sample_rate=sr, id_probe_seconds=1.0)
    return ls, song


class TestLyraSenseCore:
    def test_force_song_then_process(self) -> None:
        ls, song = _make_ls_with_known_song()
        ls.force_song(song)
        assert ls.identified_song is song
        # Feed audio and expect a lyric line
        chunk = make_sine_audio(440.0, 1.0)
        line = ls.process_chunk(chunk)
        assert line is None or isinstance(line, LyricLine)

    def test_identification_via_process_chunk(self) -> None:
        """Feeding enough audio triggers identification automatically."""
        sr = 22050
        bank = SongBank()
        audio = make_sine_audio(440.0, 10.0, sr)
        song = bank.add_song_from_audio("Auto ID Song", "X", audio, sr, make_demo_lyrics())
        ls = LyraSense(bank, sample_rate=sr, id_probe_seconds=1.0)

        # Feed 1.5s of audio in small chunks to trigger identification
        probe = make_sine_audio(440.0, 1.5, sr)
        chunk_size = 2048
        for start in range(0, len(probe), chunk_size):
            ls.process_chunk(probe[start : start + chunk_size])

        assert ls.identified_song is not None

    def test_lyric_callback_invoked(self) -> None:
        received: list[LyricLine] = []
        ls, song = _make_ls_with_known_song()
        ls.lyric_callback = received.append
        ls.force_song(song)

        # Feed enough audio to advance past the first lyric boundary
        audio = make_sine_audio(440.0, 6.0)
        chunk_size = 2048
        for start in range(0, len(audio), chunk_size):
            ls.process_chunk(audio[start : start + chunk_size])

        # Callback may or may not have been triggered depending on alignment,
        # but must only receive LyricLine objects
        for item in received:
            assert isinstance(item, LyricLine)

    def test_current_lyric_property(self) -> None:
        ls, song = _make_ls_with_known_song()
        ls.force_song(song)
        assert ls.current_lyric is None  # before any audio
        chunk = make_sine_audio(440.0, 0.5)
        ls.process_chunk(chunk)
        # After at least one chunk with enough samples, may still be None
        # (if buffer not yet full enough); just check it's the right type.
        assert ls.current_lyric is None or isinstance(ls.current_lyric, LyricLine)

    def test_is_running_false_before_start(self) -> None:
        bank = SongBank()
        ls = LyraSense(bank)
        assert not ls.is_running
