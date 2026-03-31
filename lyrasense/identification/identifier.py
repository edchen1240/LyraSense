"""Song identification from a short audio snippet.

The identifier uses cosine similarity between the chroma feature sequence of
the incoming audio and the first few seconds of each reference song in the bank.
This provides a lightweight, training-free approach suitable for a confined song
bank (tens of songs).

For larger banks, a fingerprinting approach (e.g. Dejavu / audfprint) would be
preferable; this implementation keeps the dependency footprint minimal.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from lyrasense.audio.features import FeatureExtractor
from lyrasense.song_bank.manager import SongBank
from lyrasense.song_bank.song import Song

logger = logging.getLogger(__name__)


class SongIdentifier:
    """Identify the most likely song from a short audio clip.

    Parameters
    ----------
    song_bank:
        The :class:`~lyrasense.song_bank.manager.SongBank` to search.
    extractor:
        Feature extractor shared with the rest of the pipeline (avoids
        constructing a second instance with different parameters).
    probe_seconds:
        How many seconds of audio to use as a probe for identification.
    hop_length:
        Hop length used when extracting probe features.
    """

    def __init__(
        self,
        song_bank: SongBank,
        extractor: Optional[FeatureExtractor] = None,
        probe_seconds: float = 3.0,
        hop_length: int = 512,
    ) -> None:
        self.song_bank = song_bank
        self.extractor = extractor or FeatureExtractor()
        self.probe_seconds = probe_seconds
        self.hop_length = hop_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def identify(
        self, audio: np.ndarray, sample_rate: int
    ) -> Tuple[Optional[Song], float]:
        """Identify the song matching *audio*.

        Parameters
        ----------
        audio:
            Mono audio waveform (1-D float array).
        sample_rate:
            Sample rate of *audio* in Hz.

        Returns
        -------
        (song, score)
            The best-matching :class:`~lyrasense.song_bank.song.Song` and its
            similarity score in [0, 1].  Returns ``(None, 0.0)`` when the bank
            is empty or no reference features are available.
        """
        songs = [s for s in self.song_bank.all_songs() if s.reference_features is not None]
        if not songs:
            logger.warning("No songs with reference features in bank; cannot identify.")
            return None, 0.0

        # Extract probe features from the first probe_seconds of audio
        probe_samples = int(self.probe_seconds * sample_rate)
        probe_audio = audio[:probe_samples] if len(audio) >= probe_samples else audio
        probe_feats = self.extractor.extract(probe_audio, sample_rate, hop_length=self.hop_length)

        scores: List[Tuple[Song, float]] = []
        for song in songs:
            score = self._match_score(probe_feats, song)
            scores.append((song, score))
            logger.debug("Song %r: score=%.4f", song.title, score)

        best_song, best_score = max(scores, key=lambda x: x[1])
        logger.info("Identified: %r (score=%.4f)", best_song.title, best_score)
        return best_song, best_score

    def identify_all_scores(
        self, audio: np.ndarray, sample_rate: int
    ) -> List[Tuple[Song, float]]:
        """Return similarity scores for *all* songs, sorted descending.

        Useful for debugging or UI that shows a candidate list.
        """
        songs = [s for s in self.song_bank.all_songs() if s.reference_features is not None]
        if not songs:
            return []

        probe_samples = int(self.probe_seconds * sample_rate)
        probe_audio = audio[:probe_samples] if len(audio) >= probe_samples else audio
        probe_feats = self.extractor.extract(probe_audio, sample_rate, hop_length=self.hop_length)

        scores = [(s, self._match_score(probe_feats, s)) for s in songs]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _match_score(self, probe_feats: np.ndarray, song: Song) -> float:
        """Compute a similarity score between *probe_feats* and *song*'s reference.

        We compare the probe against multiple windows in the reference (to
        handle the case where the user starts at different positions) and take
        the maximum cross-correlation score.  This also handles chorus
        repetitions naturally – any occurrence of the same audio segment will
        match.

        Score is the mean cosine similarity between matched probe frames and the
        best-aligned reference window.
        """
        ref = song.reference_features  # (R, D)
        probe = probe_feats  # (P, D)

        if ref is None or len(ref) == 0 or len(probe) == 0:
            return 0.0

        n_probe = len(probe)
        n_ref = len(ref)

        if n_ref < n_probe:
            # Reference shorter than probe – compare the whole reference
            return float(self._window_score(probe[:n_ref], ref))

        # Slide probe over reference and take the maximum score
        best = 0.0
        step = max(1, n_probe // 4)  # stride for efficiency
        for start in range(0, n_ref - n_probe + 1, step):
            window = ref[start : start + n_probe]
            s = self._window_score(probe, window)
            if s > best:
                best = s
        return best

    @staticmethod
    def _window_score(a: np.ndarray, b: np.ndarray) -> float:
        """Mean cosine similarity between two equal-length feature matrices."""
        # Both matrices are already L2-normalised (done in FeatureExtractor)
        dot = np.einsum("ij,ij->i", a, b)  # per-frame dot product
        return float(np.mean(dot))
