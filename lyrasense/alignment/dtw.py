"""Online Dynamic Time Warping (Online DTW) for real-time audio-lyric alignment.

Reference
---------
Dixon, S. (2005). "Live tracking of musical performances using on-line time
warping."  Proceedings of the 8th International Conference on Digital Audio
Effects (DAFx-05).

This implementation follows the standard Online DTW formulation: at each step
a new query frame arrives and is matched against a window of reference frames,
maintaining a running cost matrix strip.  The path is extracted greedily to
find the current reference position in amortised O(W) time per query frame
(W = window half-width).

Key design decisions
--------------------
* **Window size** W: The search window limits how far the estimated position can
  deviate from the diagonal.  A larger W handles more tempo variation but
  increases CPU cost and introduces more latency.
* **Local cost**: cosine distance (1 – cosine similarity) between L2-normalised
  feature vectors.
* **Step pattern**: unit steps {(1,0), (0,1), (1,1)} (standard DTW).
* **Current position**: the reference frame index corresponding to the latest
  accepted warping path point.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_INF = np.inf


class OnlineDTW:
    """Streaming DTW aligner.

    Parameters
    ----------
    reference:
        Full reference feature matrix, shape ``(R, D)``.  Must be
        L2-normalised (as produced by :class:`~lyrasense.audio.features.FeatureExtractor`).
    window:
        Half-width of the Sakoe–Chiba band (number of frames).  A value of
        ``max(50, R // 10)`` is a reasonable default.
    """

    def __init__(self, reference: np.ndarray, window: int = 100) -> None:
        if reference.ndim != 2:
            raise ValueError("reference must be a 2-D array (frames × features).")
        self.reference = reference
        self.window = window

        R = len(reference)
        W = window

        # Accumulated cost strip: we keep a (2W+1)-wide band around the
        # current diagonal.  Rows correspond to reference frames; we keep only
        # a sliding strip of width 2*W+1 centred on the current query frame.
        # Implementation: we store the *full* accumulated cost matrix in a
        # compact (R, 2W+1) array to avoid repeated allocation.  For very
        # long songs this could be replaced by a rolling buffer.
        self._D = np.full((R, 2 * W + 1), _INF, dtype=np.float64)

        self._n = 0          # number of query frames processed so far
        self._ref_pos: int = 0  # current estimated reference frame

        # Initialise the first column of the band (ref frames 0..W)
        # with cumulative cost of matching query frame 0 (not yet received).
        # The actual init happens lazily on the first call to step().

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, query_frame: np.ndarray) -> int:
        """Process one query feature frame and return the current reference position.

        Parameters
        ----------
        query_frame:
            1-D feature vector of length ``D``, L2-normalised.

        Returns
        -------
        int
            Estimated index into the reference sequence corresponding to the
            current playback position.
        """
        n = self._n
        R = len(self.reference)
        W = self.window
        D = self._D

        # --- Row range in reference to update ---
        r_lo = max(0, n - W)
        r_hi = min(R, n + W + 1)

        # --- Local cost: cosine distance (1 - dot product of normed vecs) ---
        ref_window = self.reference[r_lo:r_hi]          # (k, feat_dim)
        dots = ref_window @ query_frame                  # (k,)  cosine sim
        local_cost = 1.0 - np.clip(dots, -1.0, 1.0)     # (k,)

        # --- Accumulated cost update ---
        # band column index: col = r - (n - W)
        offset = n - W  # r_lo may be > 0 if n < W

        for k, r in enumerate(range(r_lo, r_hi)):
            col = r - offset           # column in the band for (n, r)
            c = local_cost[k]

            # Previous accumulated costs for the three step predecessors
            # Predecessor (n-1, r):   horizontal step
            c_top = D[r, col - 1] if (n > 0 and 0 <= col - 1 < 2 * W + 1) else _INF
            # Predecessor (n, r-1):   vertical step (within same query frame)
            c_left = D[r - 1, col] if r > 0 else _INF
            # Predecessor (n-1, r-1): diagonal step
            c_diag = D[r - 1, col - 1] if (n > 0 and r > 0 and 0 <= col - 1 < 2 * W + 1) else _INF

            D[r, col] = c + min(c_top, c_left, c_diag)

        # --- Track current reference position ---
        # Find the reference frame whose accumulated cost (at the current band
        # column diagonal entry) is smallest.  For query step n, reference
        # frame r sits at band column (r - offset).
        if r_hi > r_lo:
            diag_costs = np.array([D[r, r - offset] for r in range(r_lo, r_hi)])
            best_local = int(np.argmin(diag_costs))
            self._ref_pos = r_lo + best_local

        self._n += 1
        return self._ref_pos

    def reset(self, reference: Optional[np.ndarray] = None) -> None:
        """Reset the aligner, optionally with a new reference sequence."""
        if reference is not None:
            self.reference = reference
        R = len(self.reference)
        W = self.window
        self._D = np.full((R, 2 * W + 1), _INF, dtype=np.float64)
        self._n = 0
        self._ref_pos = 0

    @property
    def current_ref_frame(self) -> int:
        """Current estimated reference frame index."""
        return self._ref_pos

    @property
    def query_frames_processed(self) -> int:
        """Number of query frames processed so far."""
        return self._n
