"""Song data structures for LyraSense."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LyricLine:
    """A single line of lyrics with optional timing information.

    Attributes:
        text: The lyric text.
        start_time: Start time in seconds (None if unknown).
        end_time: End time in seconds (None if unknown).
    """

    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        """Duration of the lyric line in seconds."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return f"LyricLine({self.text!r}, {self.start_time}s–{self.end_time}s)"


@dataclass
class Song:
    """A song entry in the LyraSense song bank.

    Attributes:
        title: Song title.
        artist: Artist name.
        lyrics: Ordered list of lyric lines with timing.
        sample_rate: Sample rate of the reference audio (Hz).
        reference_features: Precomputed audio feature matrix (frames × features).
            Rows correspond to successive analysis frames; columns to feature bins.
        feature_hop_length: Number of audio samples between successive feature frames.
    """

    title: str
    artist: str
    lyrics: List[LyricLine] = field(default_factory=list)
    sample_rate: int = 22050
    reference_features: Optional["numpy.ndarray"] = None  # type: ignore[name-defined]
    feature_hop_length: int = 512

    def get_lyric_at_time(self, time_sec: float) -> Optional[LyricLine]:
        """Return the lyric line active at *time_sec*, or None.

        If timing information is not available, return the first line as a
        fallback so the caller always receives something displayable.
        """
        for line in self.lyrics:
            if (
                line.start_time is not None
                and line.end_time is not None
                and line.start_time <= time_sec < line.end_time
            ):
                return line
        # Fallback: return the last line whose start_time ≤ time_sec
        last: Optional[LyricLine] = None
        for line in self.lyrics:
            if line.start_time is not None and line.start_time <= time_sec:
                last = line
        return last

    def time_to_frame(self, time_sec: float) -> int:
        """Convert a time in seconds to the nearest reference feature frame index."""
        fps = self.sample_rate / self.feature_hop_length
        return max(0, round(time_sec * fps))

    def frame_to_time(self, frame_idx: int) -> float:
        """Convert a feature frame index to the corresponding time in seconds."""
        fps = self.sample_rate / self.feature_hop_length
        return frame_idx / fps

    @property
    def duration(self) -> Optional[float]:
        """Total song duration derived from the last lyric end time."""
        for line in reversed(self.lyrics):
            if line.end_time is not None:
                return line.end_time
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return f"Song({self.title!r} by {self.artist!r}, {len(self.lyrics)} lines)"
