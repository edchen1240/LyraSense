"""
[LRS_M3_DTW.py]
Purpose: 
Author: Meng-Chi Ed Chen
Date: 
Reference:
    1.
    2.

Status: Working.
"""
import os, sys, time
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime
import matplotlib.pyplot as plt


import LRS_M1_File as M1F
import LRS_M2_Data as M2D

DTW_SETTINGS = {
    # --- Similarity matrix computation (used by compute_similarity_matrix) ---
    'normalize':            True,           # Apply column-wise L2 normalization before computing distances
    'metric':               'cosine',       # Distance/similarity function: 'cosine', 'euclidean', 'manhattan'
    'smooth_len':           0,              # Temporal smoothing window width (frames); 0 or 1 to disable; typical 9–21
    'energy_threshold':     0.0,            # Silence masking threshold (L2 norm); 0.0 to disable; typical 0.005–0.05
    'use_cens':             False,          # Apply CENS quantize→smooth→re-normalize for robustness to loudness/timbre
    # --- DTW path computation ---
    'band_radius':          None,           # Sakoe–Chiba or Itakura constraint — limits search diagonal
    'step_pattern':         'symmetric2',   # Allowed transitions — controls monotonicity and slope constraints
    'subsequence_dtw':      False,          # Find a short query within a longer reference (vs. full alignment)
    'path_normalization':   False,          # Divide cumulative cost by path length for cross-pair comparability
    'reinit_threshold':     None,           # Cost threshold above which re-synchronization is triggered
    'max_jump_frames':      None            # Max positional jump during re-sync
}



class DTWMatching:
    def __init__(self, 
                 dir_REF, 
                 dir_output,
                 track_ID_1, 
                 track_ID_2,
                 DTW_run_settings):
        
        #[1] Store parameters
        self.dir_REF = dir_REF
        self.dir_output = dir_output
        self.track_ID_1 = track_ID_1
        self.track_ID_2 = track_ID_2
        
        #[2] Store DTW settings
        self.DTW_run_settings = DTW_run_settings
        self.add_DTW_path   = DTW_run_settings['add_DTW_path']
        self.key_correction = DTW_run_settings['key_correction']
        
        #[3] Load chroma features for both tracks
        self.recd_meta_1, self.chroma_1 = self.load_recd_meta_and_chroma(track_ID_1)
        self.recd_meta_2, self.chroma_2 = self.load_recd_meta_and_chroma(track_ID_2)
        self.verify_chroma_shapes()
        self.shift_chroma_key_correction()
       
        #[4] For alignment path.
        self.dtw_cost_matrix    = None
        self.dtw_path           = None
        self.dtw_total_cost     = None
        self.elapsed_path       = None
        
        #[5] For visualization and saving results.
        self.tstmp          = datetime.now().strftime('%Y-%m%d-%H%M')
        self.path_save_vis  = os.path.join(self.dir_output, f'{self.tstmp}_{self.track_ID_1}_vs_{self.track_ID_2}_cstM.png')
        self.path_save_json = os.path.join(self.dir_output, f'{self.tstmp}_.json')
        self.meta_DTW_settings       = DTW_SETTINGS.copy()
        self.elapsed_DTW    = None
    
        
    def load_recd_meta_and_chroma(self, track_ID):
        #[1] Placeholder: Load chroma features from precomputed files or compute on the fly
        work_ID, recording_ID     = track_ID.split('-')
        path_recd_meta = os.path.join(self.dir_REF, work_ID, recording_ID, f'{recording_ID}_recording meta.json')
        recd_meta = M1F.read_json_as_dict(path_recd_meta)
        
        #[4] Load chroma features from path specified in recd_meta.
        path_chroma = recd_meta['c_path_chroma']
        if not os.path.exists(path_chroma):
            raise FileNotFoundError(f'\nChroma features not found for {track_ID} at {path_chroma}')
        # Load npz file at path_chroma and convert to np array.
        chroma = np.load(path_chroma)['chroma'] 
        
        
        return recd_meta, chroma
        
    def verify_chroma_shapes(self):
        print(f'\n[verify_chroma_shapes]')
        cnt_F1, cnt_T1 = self.chroma_1.shape
        cnt_F2, cnt_T2 = self.chroma_2.shape
        diff_ratio_T = abs(cnt_T1 - cnt_T2)/max(cnt_T1, cnt_T2)
        print(f'Chroma shape for {self.track_ID_1}: {self.chroma_1.shape}')
        print(f'Chroma shape for {self.track_ID_2}: {self.chroma_2.shape}')
        
        if cnt_F1 != cnt_F2:
            print(f'⚠️ Warning: Number of chroma bins differ between tracks ({cnt_F1} vs {cnt_F2}). DTW may be affected.')
        elif diff_ratio_T > 0.1:
            print(f'⚠️ Warning: Significant difference in time frames between tracks ({cnt_T1} vs {cnt_T2}; diff_ratio_T = {diff_ratio_T:.2%}). \nDTW may be affected.')
        else:
            print(f'✅ Chroma shapes are compatible for DTW matching. (diff_ratio_T = {diff_ratio_T:.2%})')


    def shift_chroma_key_correction(self):
        """
        Detect key string, compute semitone difference, and shift chroma features of track 2 to match track 1.
        """
        if not self.key_correction:
            return
        #[1] Get 'starting key' from both tracks.
        _, [key_1] = M2D.key_symbol_in_dict_valid(self.recd_meta_1, ['starting key'])
        _, [key_2] = M2D.key_symbol_in_dict_valid(self.recd_meta_2, ['starting key'])
        
        print(f'\n[key_correction] (key_1, key_2) = ({key_1}, {key_2}). Change key_2 to match key_1.')
        
        sys.exit('Waiting for key correction implementation...')
    
    
    def compute_similarity_matrix(self, dict_DIW_settings=DTW_SETTINGS):
        """
        Parameters
        ----------
        normalize : bool
            Apply column-wise L2 normalization before computing distances.
            Recommended: True.

        metric : str  {'cosine', 'euclidean', 'manhattan'}
            Choice of metric used to compute the matrix:
            - 'cosine'    : cosine similarity (higher = more similar; range ≈ -1..1)
            - 'euclidean' : L2 distance per frame pair (lower = more similar)
            - 'manhattan' : L1 distance per frame pair (lower = more similar)
            Note: 'cosine' returns a similarity measure; the other options return distances.

        smooth_len : int (odd number, or 0 to disable)
            Width of the temporal smoothing window applied to chroma before
            metric computation.  A window of 9–21 frames (~0.2–0.5 s at
            typical hop sizes) reduces local jitter and reveals global
            structure.  Set to 0 or 1 to skip smoothing.

        energy_threshold : float  (0.0 to disable)
            Frames whose L2 norm (before normalization) are below this
            threshold are considered silent and will be masked to the 'worst'
            value for the chosen metric (minimum for similarity metrics such
            as 'cosine', maximum for distance metrics). Typical range: 0.005 – 0.05.

        use_cens : bool
            Apply CENS-style post-processing (quantize → smooth → re-normalize)
            before distance computation.  CENS features are more robust to
            tempo, dynamics, and timbre variations.
            Recommended when recordings differ significantly in loudness or
            articulation.
        """
        print(f'\n[compute_similarity_matrix]')
        t1 = time.time()
        #[1] Check if all required keys.
        required_keys = ['normalize', 'metric', 'smooth_len', 'energy_threshold', 'use_cens']
        missing_keys = [k for k in required_keys if k not in dict_DIW_settings]
        if missing_keys:
            print(f"⚠️ [Warning] Missing DTW settings: {missing_keys}")

        #[1] Copy chroma to avoid modifying original
        C1 = self.chroma_1.copy()   # (12, T1)
        C2 = self.chroma_2.copy()   # (12, T2)

        #[2] Record per-frame energy BEFORE any normalization (used for masking)
        energy_1 = np.linalg.norm(C1, axis=0)  # (T1,)
        energy_2 = np.linalg.norm(C2, axis=0)  # (T2,)

        #[3] Optional L2 normalization
        if dict_DIW_settings['normalize']:
            C1 = C1 / (np.linalg.norm(C1, axis=0, keepdims=True) + 1e-8)
            C2 = C2 / (np.linalg.norm(C2, axis=0, keepdims=True) + 1e-8)
            print('-- Applied column-wise L2 normalization.')

        #[4] Optional CENS-style post-processing
        if dict_DIW_settings['use_cens']:
            #[4a] Quantize with a log-style mapping (5 bins)
            for C in (C1, C2):
                C[C < 0.05] = 0.0
                C[(C >= 0.05) & (C < 0.1)] = 1.0
                C[(C >= 0.1)  & (C < 0.2)] = 2.0
                C[(C >= 0.2)  & (C < 0.4)] = 3.0
                C[C >= 0.4] = 4.0
            #[4b] Re-normalize after quantization
            C1 = C1 / (np.linalg.norm(C1, axis=0, keepdims=True) + 1e-8)
            C2 = C2 / (np.linalg.norm(C2, axis=0, keepdims=True) + 1e-8)
            print('-- Applied CENS quantization + re-normalization.')

        #[5] Optional temporal smoothing (uniform moving average along time axis)
        smooth_len = dict_DIW_settings['smooth_len']
        if smooth_len and smooth_len > 1:
            kernel = np.ones(smooth_len) / smooth_len
            C1 = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, C1)
            C2 = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, C2)
            #[5a] Re-normalize after smoothing to keep unit vectors
            C1 = C1 / (np.linalg.norm(C1, axis=0, keepdims=True) + 1e-8)
            C2 = C2 / (np.linalg.norm(C2, axis=0, keepdims=True) + 1e-8)
            print(f'-- Applied temporal smoothing (window={smooth_len}).')

        #[6] Compute pairwise distance matrix
        metric = dict_DIW_settings['metric']
        if metric == 'cosine':
            # similarity: (T1, T2)
            similarity = np.dot(C1.T, C2)
            similarity_matrix = similarity
        elif metric == 'euclidean':
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b  (fast broadcast)
            sq1 = np.sum(C1 ** 2, axis=0, keepdims=True).T   # (T1, 1)
            sq2 = np.sum(C2 ** 2, axis=0, keepdims=True)      # (1, T2)
            cross = np.dot(C1.T, C2)                           # (T1, T2)
            similarity_matrix = np.sqrt(np.maximum(sq1 + sq2 - 2 * cross, 0))
        elif metric == 'manhattan':
            # Computed via loop-free broadcasting; memory-heavy for large T
            similarity_matrix = np.sum(np.abs(C1.T[:, :, None] - C2.T[None, :, :]), axis=1)
        else:
            raise ValueError(f'Unsupported metric: {metric}. Choose from "cosine", "euclidean", or "manhattan".')

        # record metric used (for plotting/labels)
        self.similarity_metric = metric

        #[7] Silence masking: mask low-energy frames to the 'worst' value for the metric
        energy_threshold = dict_DIW_settings['energy_threshold']
        if energy_threshold > 0:
            silent_1 = energy_1 < energy_threshold   # (T1,)
            silent_2 = energy_2 < energy_threshold   # (T2,)
            # for similarity metrics (higher is better) use min; for distance metrics use max
            if metric == 'cosine':
                fill_value = similarity_matrix.min()
            else:
                fill_value = similarity_matrix.max()
            similarity_matrix[silent_1, :] = fill_value
            similarity_matrix[:, silent_2] = fill_value
            n_masked = int(silent_1.sum() + silent_2.sum())
            if n_masked:
                print(f'  Masked {silent_1.sum()} silent frames in Track 1, '
                      f'{silent_2.sum()} in Track 2 (threshold={energy_threshold}).')

        self.similarity_matrix = similarity_matrix
        self.elapsed_DTW = time.time() - t1
        print(f'-- Similarity matrix computed in {self.elapsed_DTW:.2f} seconds.')
        print(f'-- Similarity matrix shape: {similarity_matrix.shape}')
        print(f'-- Similarity range: min={similarity_matrix.min():.4f}, max={similarity_matrix.max():.4f}')

        return similarity_matrix

    def compute_optimal_path(self):
        print(f'\n[compute_optimal_path]')
        
        if not hasattr(self, 'similarity_matrix'):
            raise AttributeError('Similarity matrix not computed. Call compute_similarity_matrix() first.')
        
        t1 = time.time()
        
        S = self.similarity_matrix
        metric = getattr(self, 'similarity_metric', 'cosine')
        
        #[1] Convert similarity to cost if needed ---
        if metric == 'cosine':
            # higher similarity is better → convert to cost
            D = 1.0 - S
        else:
            # already distance
            D = S.copy()
        
        T1, T2 = D.shape
        
        #[2] Initialize accumulated cost matrix ---
        C = np.full((T1, T2), np.inf)
        C[0, 0] = D[0, 0]
        
        # first column
        for i in range(1, T1):
            C[i, 0] = D[i, 0] + C[i - 1, 0]
        
        # first row
        for j in range(1, T2):
            C[0, j] = D[0, j] + C[0, j - 1]
        
        #[3] Dynamic programming recursion ---
        for i in range(1, T1):
            for j in range(1, T2):
                C[i, j] = D[i, j] + min(
                    C[i - 1, j],      # insertion
                    C[i, j - 1],      # deletion
                    C[i - 1, j - 1]   # match
                )
        
        #[4] Backtracking
        i, j = T1 - 1, T2 - 1
        path = [(i, j)]
        
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                steps = [
                    C[i - 1, j],      # up
                    C[i, j - 1],      # left
                    C[i - 1, j - 1]   # diag
                ]
                argmin = np.argmin(steps)
                
                if argmin == 0:
                    i -= 1
                elif argmin == 1:
                    j -= 1
                else:
                    i -= 1
                    j -= 1
            
            path.append((i, j))
        
        path.reverse()
        
        #[5] Save results ---
        self.dtw_cost_matrix = C
        self.dtw_path = np.array(path)
        self.dtw_total_cost = C[-1, -1]
        
        self.elapsed_path = time.time() - t1
        print(f'-- DTW path computed in {self.elapsed_path:.2f} seconds.')
        print(f'-- Path length: {len(self.dtw_path)}')
        print(f'-- Total alignment cost: {self.dtw_total_cost:.4f}')
        
        return self.dtw_path

    def visualize_similarity_matrix(self):
        print(f'\n[visualize_similarity_matrix] This may take a moment for large similarity matrices...')
        #[1] Validate info needed to add_DTW_path.
        if self.add_DTW_path:
            if self.dtw_path is None:
                raise ValueError('\nDTW path not computed yet. Call compute_optimal_path() first.')

        #[1] Check if similarity matrix is computed.
        if not hasattr(self, 'similarity_matrix'):
            raise AttributeError('Similarity matrix not computed yet. Call compute_similarity_matrix() first.')
        
        #[2] Plot settings.
        fig, ax = plt.subplots(figsize=(20, 20))
        fs_title, fs_axis, fs_tick, fs_cbar = 18, 14, 12, 12
        path_thickness = 2
        cmap = 'viridis'

        #[1] Display similarity/distance matrix as image (origin='lower' so time increases upward for track 2)
        img = ax.imshow(self.similarity_matrix.T,
                        aspect='auto',
                        origin='lower',
                        cmap=cmap,
                        interpolation='nearest')

        #[2] Colorbar
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        # label colorbar depending on metric used
        metric_label = getattr(self, 'similarity_metric', None)
        if metric_label == 'cosine':
            cbar.set_label('Cosine similarity', fontsize=fs_cbar)
        elif metric_label in ('euclidean', 'manhattan'):
            cbar.set_label(f'{metric_label.capitalize()} distance', fontsize=fs_cbar)
        else:
            cbar.set_label('Similarity / Distance', fontsize=fs_cbar)
        cbar.ax.tick_params(labelsize=fs_tick)

        #[3] Labels and title
        ax.set_xlabel(f'Track 1: {self.track_ID_1} (frames)', fontsize=fs_axis)
        ax.set_ylabel(f'Track 2: {self.track_ID_2} (frames)', fontsize=fs_axis)
        ax.tick_params(axis='both', labelsize=fs_tick)
        title = f'Similarity Matrix: {self.track_ID_1} vs {self.track_ID_2}'
        ax.set_title(title, fontsize=fs_title)

        #[4] Optionally overlay the DTW alignment path
        if self.add_DTW_path:
            # dtw_path is (N, 2): col 0 = Track1 frame index, col 1 = Track2 frame index.
            # The image was plotted as similarity_matrix.T with origin='lower',
            # so x maps to Track1 (axis-0) and y maps to Track2 (axis-1).
            ax.plot(self.dtw_path[:, 0],
                    self.dtw_path[:, 1],
                    color='red',
                    linewidth=path_thickness,
                    alpha=0.85,
                    label='DTW path')
            ax.legend(fontsize=fs_tick, loc='upper left')
            print(f'-- DTW path overlaid (path length = {len(self.dtw_path)}).')

        plt.tight_layout()
        

        #[15] Optionally save the figure
        plt.savefig(self.path_save_vis)
        plt.close()
        print(f'-- Plot saved to {self.path_save_vis}')
        self.combine_json_for_DTW_results(save_json=True)

        return fig, ax


    def combine_json_for_DTW_results(self, save_json=True):
        #[1] Add run result to DTW_run_settings.
        self.DTW_run_settings['elapsed_DTW_sec'] = self.elapsed_DTW
        self.DTW_run_settings['elapsed_path_sec'] = self.elapsed_path
        self.DTW_run_settings['dtw_path_length'] = len(self.dtw_path) if self.dtw_path is not None else None
        self.DTW_run_settings['dtw_total_cost'] = self.dtw_total_cost
        
        #[2] Combine DTW_run_settings, meta_DTW_settings, recd_meta_1, and recd_meta_2 into one dictionary.
        combined_dict = {
            'run_settings': self.DTW_run_settings,
            'DTW_settings': self.meta_DTW_settings,
            'recd_meta_1': self.recd_meta_1,
            'recd_meta_2': self.recd_meta_2
        }
        
        #[2] Optionally save to JSON
        if save_json:
            M1F.save_dict_as_json(combined_dict, self.path_save_json)
            print(f'-- Combined metadata saved to {self.path_save_json}')
        
    
        
    
    

