"""Shared test fixtures and helpers."""
from __future__ import annotations

import numpy as np
import pytest

from lyrasense.song_bank.song import LyricLine, Song


def make_sine_audio(
    frequency: float = 440.0,
    duration: float = 5.0,
    sample_rate: int = 22050,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a mono sine wave for testing."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def make_demo_lyrics() -> list[LyricLine]:
    return [
        LyricLine("First line",  start_time=0.0,  end_time=2.0),
        LyricLine("Second line", start_time=2.0,  end_time=4.0),
        LyricLine("Chorus",      start_time=4.0,  end_time=6.0),
        LyricLine("Chorus",      start_time=8.0,  end_time=10.0),
        LyricLine("Last line",   start_time=10.0, end_time=12.0),
    ]


@pytest.fixture
def sample_audio() -> np.ndarray:
    return make_sine_audio()


@pytest.fixture
def sample_rate() -> int:
    return 22050


@pytest.fixture
def demo_lyrics() -> list[LyricLine]:
    return make_demo_lyrics()
