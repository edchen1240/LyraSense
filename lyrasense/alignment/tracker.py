"""Streaming lyric tracker: wraps OnlineDTW to produce LyricLine outputs.

The tracker maintains a buffer of incoming audio chunks, periodically extracts
features, advances the DTW aligner and maps the estimated reference position
to the corresponding lyric line.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from lyrasense.alignment.dtw import OnlineDTW
from lyrasense.audio.features import FeatureExtractor
from lyrasense.song_bank.song import LyricLine, Song

logger = logging.getLogger(__name__)


class LyricTracker:
    """Stateful tracker that converts a stream of audio chunks into lyric lines.

    Parameters
    ----------
    song:
        The song being tracked.
    extractor:
        Shared feature extractor instance.
    hop_length:
        Hop size in samples used during feature extraction.  Must match the
        hop length used to build the reference features.
    dtw_window:
        Sakoe–Chiba band half-width for the Online DTW.
    n_fft:
        FFT window size for feature extraction.
    """

    def __init__(
        self,
        song: Song,
        extractor: Optional[FeatureExtractor] = None,
        hop_length: int = 512,
        dtw_window: int = 100,
        n_fft: int = 2048,
    ) -> None:
        if song.reference_features is None:
            raise ValueError(
                f"Song {song.title!r} has no reference features. "
                "Call SongBank.add_song_from_audio() to precompute them."
            )
        self.song = song
        self.extractor = extractor or FeatureExtractor()
        self.hop_length = hop_length
        self.n_fft = n_fft

        self._dtw = OnlineDTW(song.reference_features, window=dtw_window)
        self._buffer: Deque[np.ndarray] = deque()
        self._buffer_samples: int = 0
        self._current_line: Optional[LyricLine] = None
        self._last_update_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, chunk: np.ndarray) -> Optional[LyricLine]:
        """Add an audio chunk and return the currently active lyric line.

        The tracker accumulates chunks until it has enough samples to produce
        at least one feature frame (``hop_length`` samples), then runs the
        Online DTW step and maps the result to a lyric line.

        Parameters
        ----------
        chunk:
            1-D float32/float64 array of mono audio samples.

        Returns
        -------
        LyricLine or None
            The active lyric line after processing *chunk*, or None if the
            song has not been identified yet or no lyric information is
            available.
        """
        self._buffer.append(chunk)
        self._buffer_samples += len(chunk)

        # Wait until we have at least one full FFT window worth of samples
        if self._buffer_samples < self.n_fft:
            return self._current_line

        # Drain the buffer
        audio = np.concatenate(list(self._buffer))
        self._buffer.clear()
        self._buffer_samples = 0

        # Extract features – produces one or more frames
        t0 = time.monotonic()
        frames = self.extractor.extract_frame(
            audio, self.song.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )
        feat_ms = (time.monotonic() - t0) * 1000

        # Advance DTW for each frame
        t0 = time.monotonic()
        for frame in frames:
            ref_idx = self._dtw.step(frame)
        dtw_ms = (time.monotonic() - t0) * 1000

        logger.debug(
            "LyricTracker: feat=%.1fms dtw=%.1fms ref_frame=%d",
            feat_ms, dtw_ms, ref_idx,
        )

        # Map reference frame → time → lyric line
        ref_time = self.song.frame_to_time(ref_idx)
        self._current_line = self.song.get_lyric_at_time(ref_time)
        self._last_update_time = time.monotonic()

        return self._current_line

    def reset(self) -> None:
        """Reset the tracker to start from the beginning of the song."""
        self._dtw.reset()
        self._buffer.clear()
        self._buffer_samples = 0
        self._current_line = None

    @property
    def current_lyric(self) -> Optional[LyricLine]:
        """The last lyric line emitted by the tracker."""
        return self._current_line

    @property
    def current_ref_time(self) -> float:
        """Estimated playback time in seconds within the reference song."""
        return self.song.frame_to_time(self._dtw.current_ref_frame)

    @property
    def current_lyric_index(self) -> Optional[int]:
        """0-based index of the current lyric line in ``song.lyrics``, or None."""
        if self._current_line is None:
            return None
        try:
            return self.song.lyrics.index(self._current_line)
        except ValueError:
            return None

    def upcoming_lyrics(self, n: int = 3) -> List[LyricLine]:
        """Return the next *n* lyric lines after the current position."""
        idx = self.current_lyric_index
        if idx is None:
            return self.song.lyrics[:n]
        start = idx + 1
        return self.song.lyrics[start : start + n]
