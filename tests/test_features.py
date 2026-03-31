"""Tests for audio feature extraction."""
from __future__ import annotations

import numpy as np
import pytest

from lyrasense.audio.features import FeatureExtractor
from tests.conftest import make_sine_audio


class TestFeatureExtractor:
    SR = 22050
    HOP = 512
    N_FFT = 2048

    def test_chroma_shape(self) -> None:
        audio = make_sine_audio(duration=2.0, sample_rate=self.SR)
        extractor = FeatureExtractor(use_chroma=True, use_mfcc=False)
        features = extractor.extract(audio, self.SR, hop_length=self.HOP, n_fft=self.N_FFT)
        n_frames = max(1, (len(audio) - self.N_FFT) // self.HOP + 1)
        assert features.ndim == 2
        assert features.shape[1] == 12
        # librosa uses center=True by default (pads n_fft//2 on each side),
        # which produces ceil(N/hop) frames rather than (N-n_fft)//hop+1.
        # Allow a generous tolerance to cover both counting conventions.
        assert abs(features.shape[0] - n_frames) <= self.N_FFT // self.HOP + 1

    def test_mfcc_shape(self) -> None:
        audio = make_sine_audio(duration=2.0, sample_rate=self.SR)
        extractor = FeatureExtractor(use_chroma=False, use_mfcc=True, n_mfcc=13)
        features = extractor.extract(audio, self.SR, hop_length=self.HOP, n_fft=self.N_FFT)
        assert features.ndim == 2
        assert features.shape[1] == 13

    def test_combined_features(self) -> None:
        audio = make_sine_audio(duration=2.0, sample_rate=self.SR)
        extractor = FeatureExtractor(use_chroma=True, use_mfcc=True, n_mfcc=13)
        features = extractor.extract(audio, self.SR, hop_length=self.HOP, n_fft=self.N_FFT)
        assert features.shape[1] == 12 + 13
        assert extractor.n_features == 12 + 13

    def test_l2_normalised(self) -> None:
        """Every row should have unit L2-norm (or zero)."""
        audio = make_sine_audio(duration=2.0, sample_rate=self.SR)
        extractor = FeatureExtractor(use_chroma=True, use_mfcc=False)
        features = extractor.extract(audio, self.SR, hop_length=self.HOP)
        norms = np.linalg.norm(features, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_short_audio(self) -> None:
        """Audio shorter than n_fft should still produce at least one frame."""
        audio = make_sine_audio(duration=0.05, sample_rate=self.SR)  # ~1100 samples
        extractor = FeatureExtractor()
        features = extractor.extract(audio, self.SR, hop_length=self.HOP, n_fft=self.N_FFT)
        assert features.shape[0] >= 1

    def test_raises_if_no_feature_type(self) -> None:
        with pytest.raises(ValueError):
            FeatureExtractor(use_chroma=False, use_mfcc=False)

    def test_extract_frame_consistent(self) -> None:
        """extract_frame and extract should return the same result."""
        audio = make_sine_audio(duration=1.0, sample_rate=self.SR)
        extractor = FeatureExtractor()
        f1 = extractor.extract(audio, self.SR, hop_length=self.HOP)
        f2 = extractor.extract_frame(audio, self.SR, hop_length=self.HOP)
        np.testing.assert_array_equal(f1, f2)
