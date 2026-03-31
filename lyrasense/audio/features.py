"""Audio feature extraction for LyraSense.

Features used:
  - **Chroma** (12-bin pitch class profile): pitch-based, robust to timbre variation
    and to tempo changes – ideal for music alignment.
  - **MFCC** (first 13 coefficients): provides complementary timbral information
    useful for song identification.

Both feature sets are L2-normalised per frame before use.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import librosa  # type: ignore
    _LIBROSA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIBROSA_AVAILABLE = False


class FeatureExtractor:
    """Extract and normalise audio features from a waveform array.

    Parameters
    ----------
    n_chroma:
        Number of chroma bins (default 12 covers all 12 pitch classes).
    n_mfcc:
        Number of MFCC coefficients to compute.
    use_chroma:
        When True (default) extract chroma features.
    use_mfcc:
        When True extract MFCC features and concatenate with chroma.
    """

    def __init__(
        self,
        n_chroma: int = 12,
        n_mfcc: int = 13,
        use_chroma: bool = True,
        use_mfcc: bool = False,
    ) -> None:
        self.n_chroma = n_chroma
        self.n_mfcc = n_mfcc
        self.use_chroma = use_chroma
        self.use_mfcc = use_mfcc

        if not use_chroma and not use_mfcc:
            raise ValueError("At least one of use_chroma or use_mfcc must be True.")

    @property
    def n_features(self) -> int:
        """Dimensionality of the output feature vector."""
        dim = 0
        if self.use_chroma:
            dim += self.n_chroma
        if self.use_mfcc:
            dim += self.n_mfcc
        return dim

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int = 512,
        n_fft: int = 2048,
    ) -> np.ndarray:
        """Compute features for a full waveform.

        Parameters
        ----------
        audio:
            Mono audio waveform (1-D float array).
        sample_rate:
            Sample rate in Hz.
        hop_length:
            Hop size in samples between successive frames.
        n_fft:
            FFT window length in samples.

        Returns
        -------
        numpy.ndarray, shape (n_frames, n_features)
            L2-normalised feature matrix.
        """
        if not _LIBROSA_AVAILABLE:
            return self._fallback_extract(audio, sample_rate, hop_length, n_fft)

        parts: list[np.ndarray] = []

        if self.use_chroma:
            chroma = librosa.feature.chroma_stft(
                y=audio.astype(np.float32),
                sr=sample_rate,
                n_chroma=self.n_chroma,
                n_fft=n_fft,
                hop_length=hop_length,
            )  # shape: (n_chroma, n_frames)
            parts.append(chroma)

        if self.use_mfcc:
            mfcc = librosa.feature.mfcc(
                y=audio.astype(np.float32),
                sr=sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length,
            )  # shape: (n_mfcc, n_frames)
            parts.append(mfcc)

        features = np.vstack(parts).T  # (n_frames, n_features)
        return self._normalize(features)

    def extract_frame(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        hop_length: int = 512,
        n_fft: int = 2048,
    ) -> np.ndarray:
        """Compute features for a short audio chunk (one or more frames).

        Thin wrapper around :meth:`extract` intended for streaming use where
        *audio_chunk* contains exactly enough samples to produce at least one
        feature frame.

        Returns
        -------
        numpy.ndarray, shape (n_frames, n_features)
        """
        return self.extract(audio_chunk, sample_rate, hop_length=hop_length, n_fft=n_fft)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fallback_extract(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int,
        n_fft: int,
    ) -> np.ndarray:
        """Minimal STFT-based chroma extraction without librosa.

        This fallback is used when librosa is not available (e.g. in lightweight
        test environments).  It computes a simple 12-bin chroma from the STFT
        magnitude and is *not* intended for production use.
        """
        # Compute STFT magnitude
        n_frames = max(1, (len(audio) - n_fft) // hop_length + 1)
        n_bins = n_fft // 2 + 1

        # Use scipy.fft if available, otherwise numpy.fft
        try:
            from scipy.fft import rfft  # type: ignore
        except ImportError:
            from numpy.fft import rfft  # type: ignore

        window = np.hanning(n_fft)
        mag = np.zeros((n_frames, n_bins), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start : start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            mag[i] = np.abs(rfft(frame * window))

        # Map frequency bins to chroma bins
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        chroma = np.zeros((n_frames, self.n_chroma), dtype=np.float32)
        for b, f in enumerate(freqs[1:], start=1):
            if f <= 0:
                continue
            # MIDI note number then pitch class
            midi = 12 * np.log2(f / 440.0) + 69
            pc = int(round(midi)) % self.n_chroma
            chroma[:, pc] += mag[:, b]

        return self._normalize(chroma)

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        """L2-normalise each row; zero vectors remain zero."""
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return features / norms
