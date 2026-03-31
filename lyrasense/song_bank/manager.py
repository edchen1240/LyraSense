"""Song bank: storage and management of known songs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from lyrasense.audio.features import FeatureExtractor
from lyrasense.song_bank.song import LyricLine, Song

logger = logging.getLogger(__name__)


class SongBank:
    """Collection of :class:`Song` objects with precomputed reference features.

    Songs can be loaded from JSON descriptor files or added programmatically.

    Example JSON format::

        {
            "title": "Bohemian Rhapsody",
            "artist": "Queen",
            "sample_rate": 22050,
            "feature_hop_length": 512,
            "lyrics": [
                {"text": "Is this the real life?", "start_time": 0.0, "end_time": 4.2},
                {"text": "Is this just fantasy?",  "start_time": 4.2, "end_time": 8.1}
            ],
            "reference_features": "path/to/features.npy"
        }

    ``reference_features`` may be either a path to a ``.npy`` file relative to
    the JSON or an inline 2-D list (frames × feature_bins).
    """

    def __init__(self) -> None:
        self._songs: Dict[str, Song] = {}
        self._extractor = FeatureExtractor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_song(self, song: Song) -> None:
        """Add an already-constructed :class:`Song` to the bank."""
        key = self._key(song.title, song.artist)
        self._songs[key] = song
        logger.debug("Added song %r to bank.", key)

    def add_song_from_file(self, json_path: Union[str, Path]) -> Song:
        """Load a song from a JSON descriptor file and add it to the bank.

        Parameters
        ----------
        json_path:
            Path to the song's JSON descriptor.

        Returns
        -------
        Song
            The constructed :class:`Song` object.
        """
        json_path = Path(json_path)
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)

        lyrics = [
            LyricLine(
                text=entry["text"],
                start_time=entry.get("start_time"),
                end_time=entry.get("end_time"),
            )
            for entry in data.get("lyrics", [])
        ]

        ref_features = self._load_features(data.get("reference_features"), json_path.parent)

        song = Song(
            title=data["title"],
            artist=data.get("artist", "Unknown"),
            lyrics=lyrics,
            sample_rate=int(data.get("sample_rate", 22050)),
            reference_features=ref_features,
            feature_hop_length=int(data.get("feature_hop_length", 512)),
        )
        self.add_song(song)
        return song

    def add_song_from_audio(
        self,
        title: str,
        artist: str,
        audio: np.ndarray,
        sample_rate: int,
        lyrics: List[LyricLine],
        feature_hop_length: int = 512,
    ) -> Song:
        """Build a song entry from a raw audio array and add it to the bank.

        Parameters
        ----------
        title:
            Song title.
        artist:
            Artist name.
        audio:
            Mono audio waveform as a 1-D float32/float64 NumPy array.
        sample_rate:
            Sample rate of *audio* in Hz.
        lyrics:
            Lyric lines with timing information.
        feature_hop_length:
            Hop length in samples for the feature extractor.
        """
        features = self._extractor.extract(audio, sample_rate, hop_length=feature_hop_length)
        song = Song(
            title=title,
            artist=artist,
            lyrics=lyrics,
            sample_rate=sample_rate,
            reference_features=features,
            feature_hop_length=feature_hop_length,
        )
        self.add_song(song)
        return song

    def get(self, title: str, artist: str = "") -> Optional[Song]:
        """Retrieve a song by title (and optionally artist)."""
        key = self._key(title, artist)
        return self._songs.get(key)

    def all_songs(self) -> List[Song]:
        """Return all songs in the bank."""
        return list(self._songs.values())

    def __len__(self) -> int:
        return len(self._songs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SongBank({len(self)} songs)"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(title: str, artist: str) -> str:
        return f"{title.lower()}|{artist.lower()}"

    @staticmethod
    def _load_features(
        ref: Optional[object], base_dir: Path
    ) -> Optional[np.ndarray]:
        """Load reference features from a path or inline list."""
        if ref is None:
            return None
        if isinstance(ref, str):
            feat_path = base_dir / ref
            if feat_path.exists():
                return np.load(str(feat_path))
            logger.warning("Reference features file %s not found.", feat_path)
            return None
        # Inline list
        return np.array(ref, dtype=np.float32)
