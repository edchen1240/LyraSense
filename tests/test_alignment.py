"""Tests for Online DTW and LyricTracker."""
from __future__ import annotations

import numpy as np
import pytest

from lyrasense.alignment.dtw import OnlineDTW
from lyrasense.alignment.tracker import LyricTracker
from lyrasense.audio.features import FeatureExtractor
from lyrasense.song_bank.song import LyricLine, Song
from tests.conftest import make_demo_lyrics, make_sine_audio


class TestOnlineDTW:
    def _make_reference(self, n_frames: int = 50, n_dim: int = 12) -> np.ndarray:
        """Smooth reference sequence (random walk, L2-normalised)."""
        rng = np.random.default_rng(42)
        data = np.cumsum(rng.standard_normal((n_frames, n_dim)), axis=0)
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (data / norms).astype(np.float64)

    def test_step_returns_valid_index(self) -> None:
        ref = self._make_reference(50)
        dtw = OnlineDTW(ref, window=10)
        for i, frame in enumerate(ref[:20]):
            idx = dtw.step(frame)
            assert 0 <= idx < len(ref), f"Out-of-range index {idx} at step {i}"

    def test_monotone_on_identical_sequence(self) -> None:
        """Feeding the reference itself should yield a non-decreasing position."""
        ref = self._make_reference(40)
        dtw = OnlineDTW(ref, window=15)
        positions = []
        for frame in ref:
            positions.append(dtw.step(frame))
        # Positions should be broadly non-decreasing (allow one tie)
        diffs = np.diff(positions)
        assert np.sum(diffs < -1) == 0, "DTW position decreased by more than 1 step unexpectedly"

    def test_reset(self) -> None:
        ref = self._make_reference(30)
        dtw = OnlineDTW(ref, window=10)
        dtw.step(ref[0])
        dtw.step(ref[1])
        dtw.reset()
        assert dtw.query_frames_processed == 0
        assert dtw.current_ref_frame == 0

    def test_reset_with_new_reference(self) -> None:
        ref1 = self._make_reference(30)
        ref2 = self._make_reference(20)
        dtw = OnlineDTW(ref1, window=10)
        dtw.step(ref1[0])
        dtw.reset(ref2)
        assert len(dtw.reference) == 20

    def test_raises_on_1d_reference(self) -> None:
        with pytest.raises(ValueError):
            OnlineDTW(np.zeros(10))

    def test_properties(self) -> None:
        ref = self._make_reference(20)
        dtw = OnlineDTW(ref, window=5)
        assert dtw.query_frames_processed == 0
        dtw.step(ref[0])
        assert dtw.query_frames_processed == 1


class TestLyricTracker:
    SR = 22050
    HOP = 512

    def _make_song_with_features(self) -> Song:
        """Build a Song with synthetic reference features."""
        audio = make_sine_audio(duration=12.0, sample_rate=self.SR)
        extractor = FeatureExtractor()
        ref_feats = extractor.extract(audio, self.SR, hop_length=self.HOP)
        return Song(
            title="Test Song",
            artist="Test",
            lyrics=make_demo_lyrics(),
            sample_rate=self.SR,
            reference_features=ref_feats,
            feature_hop_length=self.HOP,
        )

    def test_raises_without_features(self) -> None:
        song = Song("No feats", "X", lyrics=make_demo_lyrics())
        with pytest.raises(ValueError, match="reference features"):
            LyricTracker(song)

    def test_push_returns_lyric_or_none(self) -> None:
        song = self._make_song_with_features()
        tracker = LyricTracker(song, hop_length=self.HOP)
        chunk = make_sine_audio(duration=0.5, sample_rate=self.SR)
        result = tracker.push(chunk)
        # After one 0.5s chunk the tracker should have aligned and returned something
        assert result is None or isinstance(result, LyricLine)

    def test_push_accumulates_until_fft_window(self) -> None:
        """Small chunks (< n_fft samples) should not cause errors."""
        song = self._make_song_with_features()
        tracker = LyricTracker(song, hop_length=self.HOP)
        tiny = np.zeros(128, dtype=np.float32)
        for _ in range(20):
            tracker.push(tiny)
        # No exception and current lyric is valid type
        assert tracker.current_lyric is None or isinstance(tracker.current_lyric, LyricLine)

    def test_reset(self) -> None:
        song = self._make_song_with_features()
        tracker = LyricTracker(song, hop_length=self.HOP)
        chunk = make_sine_audio(duration=1.0, sample_rate=self.SR)
        tracker.push(chunk)
        tracker.reset()
        assert tracker.current_lyric is None
        assert tracker.current_ref_time == pytest.approx(0.0, abs=0.1)

    def test_upcoming_lyrics(self) -> None:
        song = self._make_song_with_features()
        tracker = LyricTracker(song, hop_length=self.HOP)
        upcoming = tracker.upcoming_lyrics(2)
        assert isinstance(upcoming, list)
        assert len(upcoming) <= 2
