"""LyraSense core orchestrator.

Brings together audio capture, song identification and real-time lyric
tracking into a single high-level API.

Lifecycle
---------
1. Call :meth:`start` to begin microphone capture.
2. A short identification phase buffers audio and identifies the song.
3. Tracking begins immediately after identification.
4. Query :attr:`current_lyric` at any time or register a *lyric_callback*.
5. Call :meth:`stop` to terminate capture.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

from lyrasense.audio.capture import AudioCapture
from lyrasense.audio.features import FeatureExtractor
from lyrasense.alignment.tracker import LyricTracker
from lyrasense.identification.identifier import SongIdentifier
from lyrasense.song_bank.manager import SongBank
from lyrasense.song_bank.song import LyricLine, Song

logger = logging.getLogger(__name__)


class LyraSense:
    """Real-time lyrics-following system.

    Parameters
    ----------
    song_bank:
        The :class:`~lyrasense.song_bank.manager.SongBank` to identify from.
    sample_rate:
        Capture sample rate.  22 050 Hz is recommended (matches librosa default).
    chunk_samples:
        Number of samples per audio chunk delivered from the microphone.
        Smaller values reduce latency; 2048 ≈ 93 ms at 22 050 Hz.
    hop_length:
        Feature extraction hop size.  Must match what was used to build
        reference features.
    dtw_window:
        Sakoe–Chiba band half-width in frames for Online DTW.
    id_probe_seconds:
        Seconds of audio to collect before attempting song identification.
    lyric_callback:
        Optional callable ``fn(line: LyricLine)`` invoked whenever the active
        lyric line changes.
    device:
        sounddevice input device (None = system default).
    """

    def __init__(
        self,
        song_bank: SongBank,
        sample_rate: int = 22050,
        chunk_samples: int = 2048,
        hop_length: int = 512,
        dtw_window: int = 100,
        id_probe_seconds: float = 3.0,
        lyric_callback: Optional[Callable[[LyricLine], None]] = None,
        device: Optional[object] = None,
    ) -> None:
        self.song_bank = song_bank
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.dtw_window = dtw_window
        self.id_probe_seconds = id_probe_seconds
        self.lyric_callback = lyric_callback

        self._extractor = FeatureExtractor()
        self._identifier = SongIdentifier(
            song_bank,
            extractor=self._extractor,
            probe_seconds=id_probe_seconds,
            hop_length=hop_length,
        )

        self._capture = AudioCapture(
            sample_rate=sample_rate,
            chunk_samples=chunk_samples,
            device=device,
            callback=self._on_chunk,
        )

        # State
        self._identified_song: Optional[Song] = None
        self._tracker: Optional[LyricTracker] = None
        self._current_lyric: Optional[LyricLine] = None
        self._id_buffer: list[np.ndarray] = []
        self._id_buffer_samples: int = 0
        self._id_probe_samples: int = int(id_probe_seconds * sample_rate)
        self._lock = threading.Lock()
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start microphone capture and begin the identification–tracking loop."""
        if self._running:
            logger.warning("LyraSense is already running.")
            return
        self._running = True
        self._capture.start()
        logger.info("LyraSense started.  Listening for song identification…")

    def stop(self) -> None:
        """Stop capture and tracking."""
        self._running = False
        self._capture.stop()
        logger.info("LyraSense stopped.")

    def process_chunk(self, chunk: np.ndarray) -> Optional[LyricLine]:
        """Feed a raw audio chunk directly (for offline / testing use).

        This bypasses the microphone capture and can be used to simulate
        real-time processing on pre-recorded audio.

        Parameters
        ----------
        chunk:
            Mono float32/float64 audio samples at :attr:`sample_rate`.

        Returns
        -------
        LyricLine or None
        """
        return self._on_chunk(chunk)

    def force_song(self, song: Song) -> None:
        """Skip identification and immediately begin tracking *song*.

        Useful when the current song is known in advance.
        """
        with self._lock:
            self._identified_song = song
            self._tracker = LyricTracker(
                song,
                extractor=self._extractor,
                hop_length=self.hop_length,
                dtw_window=self.dtw_window,
            )
            self._id_buffer.clear()
            self._id_buffer_samples = 0
            logger.info("Forced tracking of %r.", song.title)

    @property
    def current_lyric(self) -> Optional[LyricLine]:
        """The currently active lyric line, or None."""
        return self._current_lyric

    @property
    def identified_song(self) -> Optional[Song]:
        """The song that was identified, or None if still in probing phase."""
        return self._identified_song

    @property
    def is_running(self) -> bool:
        """True while capture is active."""
        return self._running

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _on_chunk(self, chunk: np.ndarray) -> Optional[LyricLine]:
        """Process one audio chunk.  Called from the audio capture thread."""
        with self._lock:
            if self._tracker is None:
                # Still in identification phase: buffer audio
                self._id_buffer.append(chunk)
                self._id_buffer_samples += len(chunk)

                if self._id_buffer_samples >= self._id_probe_samples:
                    probe = np.concatenate(self._id_buffer)
                    self._id_buffer.clear()
                    self._id_buffer_samples = 0
                    self._attempt_identification(probe)
                return self._current_lyric

            # Tracking phase
            line = self._tracker.push(chunk)
            if line is not self._current_lyric:
                self._current_lyric = line
                if line is not None and self.lyric_callback is not None:
                    try:
                        self.lyric_callback(line)
                    except Exception:  # noqa: BLE001
                        logger.exception("Error in lyric_callback")

        return self._current_lyric

    def _attempt_identification(self, probe: np.ndarray) -> None:
        """Try to identify the song from *probe*; start tracking if successful."""
        song, score = self._identifier.identify(probe, self.sample_rate)
        if song is None:
            logger.warning("Identification failed – no songs in bank.")
            return

        logger.info(
            "Song identified: %r by %r (score=%.4f)",
            song.title,
            song.artist,
            score,
        )
        self._identified_song = song
        self._tracker = LyricTracker(
            song,
            extractor=self._extractor,
            hop_length=self.hop_length,
            dtw_window=self.dtw_window,
        )
        # Feed the probe audio through the tracker so we don't lose time
        probe_chunk_size = self.hop_length * 4
        for start in range(0, len(probe), probe_chunk_size):
            self._tracker.push(probe[start : start + probe_chunk_size])
        line = self._tracker.current_lyric
        if line is not None:
            self._current_lyric = line
            if self.lyric_callback is not None:
                try:
                    self.lyric_callback(line)
                except Exception:  # noqa: BLE001
                    logger.exception("Error in lyric_callback")
