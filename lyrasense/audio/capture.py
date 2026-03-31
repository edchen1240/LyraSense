"""Real-time audio capture using sounddevice.

AudioCapture records from the default microphone in a background thread and
exposes chunks via a callback or a blocking :meth:`read` call.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np

try:
    import sounddevice as sd  # type: ignore
    _SD_AVAILABLE = True
except (ImportError, OSError):  # pragma: no cover
    _SD_AVAILABLE = False

logger = logging.getLogger(__name__)


class AudioCapture:
    """Non-blocking microphone capture.

    Parameters
    ----------
    sample_rate:
        Desired capture sample rate in Hz.
    chunk_samples:
        Number of samples per chunk delivered to the callback / queue.
    device:
        sounddevice device index or name.  ``None`` uses the system default.
    callback:
        Optional callable ``fn(chunk: np.ndarray)`` invoked for each chunk in
        the audio thread.  Use this for lowest latency.  If *None*, chunks are
        queued and retrievable via :meth:`read`.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        chunk_samples: int = 2048,
        device: Optional[object] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.device = device
        self._user_callback = callback
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._stream: Optional[object] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the audio stream and begin capturing."""
        if not _SD_AVAILABLE:
            raise RuntimeError(
                "sounddevice is not available.  Install it with: pip install sounddevice"
            )
        if self._running:
            return

        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_samples,
            device=self.device,
            callback=self._sd_callback,
        )
        self._stream.start()
        logger.info(
            "AudioCapture started: sample_rate=%d, chunk=%d samples",
            self.sample_rate,
            self.chunk_samples,
        )

    def stop(self) -> None:
        """Stop capturing and close the audio stream."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("AudioCapture stopped.")

    def read(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Return the next available audio chunk or None if timed out.

        Only useful when no *callback* was provided at construction time.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        """True while the capture stream is active."""
        return self._running

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        if status:
            logger.warning("AudioCapture status: %s", status)
        chunk = indata[:, 0].copy()  # take channel 0; flatten to 1-D

        if self._user_callback is not None:
            try:
                self._user_callback(chunk)
            except Exception:  # noqa: BLE001
                logger.exception("Error in AudioCapture user callback")
        else:
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                # Drop the oldest chunk to avoid unbounded memory growth
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(chunk)
