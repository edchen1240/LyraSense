"""
[LRS_M2_Data.py]
Purpose: LyraSense: Real-Time Audio-Lyric Alignment for Live Music Performance
LyraSense is a real-time lyrics-following system aim to follow the song audio input with lyric text. 
Given a live audio stream containing singing or instrumental accompaniment, 
the system identifies the underlying song from a confined song bank and continuously aligns 
the incoming audio to lyric lines with low latency. Rather than performing full speech transcription, 
yraSense formulates the task as a streaming audio–text alignment problem, 
enabling robust tracking even in the presence of repeated song structures such as choruses. 
The system is designed to operate under soft real-time constraints, 
with end-to-end latency below one second on commodity hardware. 
The result shows that LyraSense achieves accurate line-level lyric tracking with low temporal delay, 
demonstrating the feasibility of real-time lyrics following in live, informal music settings.

Author: Meng-Chi Ed Chen
Date: 
Reference:
    1.
    2.

Status: Working.
"""
import os, sys, librosa, json
import soundfile as sf
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime
import matplotlib.pyplot as plt


import LRS_M1_File as M1F




# ============================================================
# [1] LyraSense Global Settings - Start
# ============================================================

#[1] Audio & Sampling
expect_sr       = 44100   # Target sampling rate for all processing (Hz)
resample_audio  = True    # Whether to resample input to expect_sr

#[2] STFT Parameters
n_fft           = 2048    # Number of FFT points (frequency resolution)
win_length      = n_fft   # Window length for STFT
hop_length      = n_fft // 4  # 75% overlap for stable temporal tracking
window_type     = "hann"  # Window function to reduce spectral leakage
center_frames   = True    # Pad signal so frames are centered

#[3] Derived quantity (important for time mapping)
frame_duration  = hop_length / expect_sr  # Seconds per frame

#[4] Chroma Feature Parameters
feature_type        = "chroma_cqt"  # "chroma_stft", "chroma_cqt", or "chroma_cens"
n_chroma            = 12            # Pitch classes (Western equal temperament)
fmin                = 32.7          # Minimum frequency (C1) to suppress sub-bass noise
fmax                = 8000          # Maximum frequency to avoid excessive high-frequency noise
normalize_feature  = True          # L2 normalize each frame for DTW stability

#[5] DTW Alignment Parameters
dtw_metric          = "cosine"  # Distance metric ("euclidean", "cosine")
dtw_band_radius     = 50        # Sakoe–Chiba band radius (frames) for real-time pruning
dtw_step_pattern    = "symmetric2"  # Constrained step pattern for monotonic alignment

#[6] Streaming / Belief Control
min_confidence_sec      = 0.2  # Require stable alignment before lyric switch
min_confidence_frames   = int(min_confidence_sec / frame_duration)
reinit_threshold        = 0.3  # DTW cost threshold for triggering re-synchronization
max_jump_sec            = 5.0  # Maximum allowed jump when re-aligning (seconds)

#[7] Meata Data Templates
WORK_META_TEMPLATE = {
    'work_ID':              '',
    'work year':            None,
    'canonical title':      '',
    'original artist':      '',
    'language':             '',
    'oa type':              '',
    'starting key':         '',
    'key modulation':       None,
    'source':               '',
    'file name':            '',
    'path original audio':  '',
}

RECD_META_TEMPLATE = {
    #[1] Basic.
    'track_ID':             '',
    'work_ID':              '',
    'recording_ID':         '',
    'release year':         None,
    'canonical title':      '',
    'recording artist':     '',
    'source':               '',
    'file name':            '',
    'path recording audio': '',
    'original version':     None,
    'official source':      None,   
    'trim music':           None,
    #[2] Audio.
    'a_filesize_kB':        None,
    'a_n_channels':         None,
    'a_sampling_rate':      None,
    'a_n_samples':          None,
    'a_duration_sec':       None,
    'a_duration_min':       None,
    'a_value_limits':       None,
    'a_value_cnts':         None,
    #[3] Chroma.
    'c_path_chroma':            '',
    'c_chroma_kB':              None,
    'c_chroma_shape':           None,
    'hop_length':               None,
    'feature_type':             None,
    'n_chroma':                 None,
    'bins_per_octave':          None,
    'fmin':                     None,
    'n_octaves':                None,
    'apply_log_compression':    None,
    'apply_smoothing':          None,
    'smoothing_window':         None,
    'apply_cens_quantization':  None,
    'normalize_feature':        None,
    'energy_threshold':         None,
    'feature_version':          None,
}
    
    
    
    
    
    
    
    
    
    
    
CHROMA_SETTINGS = {
    # [A] Time Resolution: Controls temporal granularity of alignment
    'hop_length': 1024,             # 512–2048
                                    # 512  ≈ 11.6 ms  (very fine, jittery)
                                    # 1024 ≈ 23 ms    (recommended start)
                                    # 2048 ≈ 46 ms    (more structural)
    # [B] Chroma Type
    'feature_type': 'chroma_cqt',   # stft → faster, but less pitch-stable
                                    # cqt  → better pitch stability
                                    # cens → best structural robustness
    'n_chroma': 12,                 # 12 (standard Western)
                                    # 24 (if microtonal refinement later)
    'bins_per_octave': 36,          # 24–48
                                    # Higher → better pitch resolution
                                    # 36 is a good compromise
    'fmin': 32.7,                   # C1 (suppress sub-bass noise)
                                    # 27.5–65.4 typical
    'n_octaves': 6,                 # 5–7 typical
                                    # Controls frequency coverage
    # [C] Post Processing
    'apply_log_compression': True,  # Reduces dynamic range bias

    'apply_smoothing': True,
    'smoothing_window': 21,         # 9–41 frames
                                    # ~0.2–1.0 sec depending on hop
    'apply_cens_quantization': False,  # True for tempo-robust alignment
    # [D] Normalization
    'normalize_feature': True,      # L2 normalize each frame
    'energy_threshold': 0.01,       # 0.005–0.05
                                    # Silence masking
    # [E] Experimental / Version Control
    'feature_version': 'v1.0_cqt_structural'
}




#[8] Track Table Settings
"""
Columns in df_track_list:
['Work Year', 'Release Year', 'Canonical Title', 
'Original Artist', 'Recording Artist', 'track_ID', 
'Source', 'File Name', 'original version', 'official source', 
'trim music', 'Language', 'OA Type', 'Starting Key', 'Key Modulation']
"""
SHTN_TRACK_LIST     = '01_trackList'
SHTN_TRACK_CATALOG  = '02_trackCatalog'
PATH_TRACK_TABLE    = r'D:\01_Floor\a_Ed\09_EECS\10_Python\03_Developing\2025-0906_Music Chord Embedding\MCE-02_Data\01_Track Data List.xlsx'
DIR_SONG_BANK       = r'D:\05_Datasets\05_Song Data\01_Audio Files'
DIR_REF             = r'D:\05_Datasets\05_Song Data\02_LyraSense Ref'

LIST_COL_WORK_META  = ['Work Year', 'Canonical Title', 'Original Artist', 'Source',
                       'File Name', 'Language', 'OA Type', 'Starting Key', 'Key Modulation']
LIST_COL_RECD_META   = ['Release Year', 'Canonical Title', 'Recording Artist', 'Source', 
                       'File Name', 'original version', 'official source', 'trim music']
keys_should_be_number     = ['work year', 'release year', 'original version', 'official source', 'trim music', 'key modulation']




# ============================================================
# [1] LyraSense Global Settings - End
# ============================================================




# ============================================================
# [2] Lookup and Verification Functions - Start
# ============================================================

def read_track_table(list_col_print=[]):
    df_track_list = M1F.read_excel_sheet_into_df(PATH_TRACK_TABLE, SHTN_TRACK_LIST, False)
    #[3] Optionally print the track table with selected columns for quick review.
    if len(list_col_print) > 0:
        df_track_print = df_track_list[list_col_print]
        print(f'\n[read_track_table]\n{tabulate(df_track_print, headers="keys", tablefmt="orgtbl", showindex=False)}\n')
        return df_track_list, df_track_print
    else:
        return df_track_list



def fill_dict_with_single_row_df(df, list_col_needed=None, input_dict={}):
    #[1] Check if df has exactly one row.
    if len(df) != 1:
        text_error =    f'\n ❌ [Error] The dataframe should have exactly one row, but got {len(df)} rows.\n'\
                        f'Please check the track table as well as the track_ID or work_ID you are using for lookup.\n'
        raise ValueError(text_error)
    #[2] Use all the columns in the df if list_col_needed is not provided.
    if list_col_needed is None:
        list_col_needed = list(df.columns)

    # [3] Initialize input_dict if not provided.
    if input_dict is None:
        input_dict = {}

    # [4] Fill the input_dict with key-value pairs from the single row of the df, using list_col_needed to select columns.
    row = df.iloc[0]
    for i_col in list_col_needed:
        key = str(i_col).lower()  # Force every key to lower case.
        input_dict[key] = str(row[i_col]) if pd.notna(row[i_col]) else None
    return input_dict




    
def lookup_df_track_list_by_track_ID(track_ID, df_track_list, list_col_needed=None):
    # [1] Look up the row in df_track_list (from excel) matching this track_ID.
    row = df_track_list[df_track_list['track_ID'] == track_ID]
    # [2] Use fill_dict_with_single_row_df to extract metadata fields from the matched row.
    output_meta = fill_dict_with_single_row_df(row, list_col_needed)
    return output_meta




def lookup_df_track_list_by_work_ID(work_ID, 
                                    df_track_list, 
                                    list_col_needed=None,
                                    ):
    """
    work_ID: for lookup.
    df_track_list: The table to look up.
    list_col_needed: The columns needed to fill into the output dict. If None, use all columns in df.
    """
    # [1] Look up the rows in df_track_list (from excel) matching this work_ID.
    df_rows = df_track_list[df_track_list['track_ID'].str.startswith(work_ID + '-')]
    if df_rows.empty:
        raise ValueError(f'❌ [Error] No track found in df_track_list with work_ID "{work_ID}". Please check the track table as well as the work_ID you are using for lookup.')
    else:
        #[2] Target the original version with 'original version' == 1.
        df_original = df_rows[df_rows['original version'] == 1]
        #[3] If there are multiple versions, just take the first one and print a warning.
        print(f'-- work_ID {work_ID} has {len(df_rows)} recording(s) in total, with {len(df_original)} original version.')
        if len(df_original) == 0:
            print(f'⚠️ [Warning] No original version (original version == 1) found for work_ID "{work_ID}". Just take the first version for lookup.')
            df_original = df_rows.iloc[[0]]
        
        #[5] Initialized work_meta with WORK_META_TEMPLATE.
        work_meta = WORK_META_TEMPLATE.copy()
        work_meta['work_ID'] = work_ID
        
        #[4] Use fill_dict_with_single_row_df to extract metadata fields from the matched row.
        output_meta = fill_dict_with_single_row_df(df_original, list_col_needed, work_meta)
    return output_meta


def collect_recd_ID_folders():
    """
    Under DIR_REF, collect all directory that are exactly two levels deep.
    The first level is work_ID, and the second level is recording_ID. 
    Return df_recd_folders with columns ['track_ID', 'recd_ID', 'path_recd_folder'].
    """
    list_recd_folders = []
    for work_folder in os.listdir(DIR_REF):
        path_work_folder = os.path.join(DIR_REF, work_folder)
        if os.path.isdir(path_work_folder):
            for recd_folder in os.listdir(path_work_folder):
                path_recd_folder = os.path.join(path_work_folder, recd_folder)
                if os.path.isdir(path_recd_folder):
                    track_ID = f'{work_folder}-{recd_folder}'
                    list_recd_folders.append({
                        'track_ID': track_ID,
                        'recd_ID': recd_folder,
                        'path_recd_folder': path_recd_folder})
    df_recd_folders = pd.DataFrame(list_recd_folders)
    text_report =   f'\n[collect_recd_ID_folders] Collected {len(df_recd_folders)} recording folders under {DIR_REF}:\n'\
                    f'{tabulate(df_recd_folders, headers="keys", tablefmt="orgtbl", showindex=False)}\n'
    print(text_report)
    return df_recd_folders
    






# ============================================================
# [2] Lookup and Verification Functions - End
# ============================================================








# ============================================================
# [3] Utility Function - Start
# ============================================================









def exam_audio_file(path_audio):
    """
    Load and examine an audio file. Reports duration, sampling rate, channels, value range, NaN/Inf counts, and file size.
    Returns dict with all diagnostic info.
    """
    print(f'\n[exam_audio_file] {os.path.basename(path_audio)}')
    
    #[1] Load audio file.
    audio, sr = read_audio_file_and_cvt_to_mono(path_audio, cvt_to_mono=True)
    assert audio.ndim == 1, f'\n❌ [Error] Audio should be mono after conversion, but got {audio.ndim} channels.'
    
    #[3] Analyze audio properties.
    n_channels      = audio.ndim
    n_samples       = len(audio)
    duration_sec    = n_samples / sr
    nan_count       = int(np.isnan(audio).sum())
    inf_count       = int(np.isinf(audio).sum())
    value_min       = round(float(np.nanmin(audio)), 6)
    value_max       = round(float(np.nanmax(audio)), 6)
    filesize_kB     = os.path.getsize(path_audio) / 1024

    #[4] Compile diagnostic info into a dict and print it.
    dict_audio_info = {
        'a_filesize_kB':    round(filesize_kB, 1),
        'a_n_channels':     n_channels,
        'a_sampling_rate':  sr,
        'a_n_samples':      n_samples,
        'a_duration_sec':   round(duration_sec, 2),
        'a_duration_min':   round(duration_sec / 60, 2),
        'a_value_limits':   [value_min, value_max],
        'a_value_cnts':     {'nan': nan_count, 'inf': inf_count},
    }
    
    #[5] Print the diagnostic info in a readable format.
    M1F.format_dict_beautifully(dict_audio_info)
    return dict_audio_info




def exam_chroma_file(path_chroma, path_audio):
    """
    Load and examine a chroma .npz file. Reports shape, duration, value range,
    NaN/Inf counts, normalization status, and file size.
    Returns dict with all diagnostic info.
    """
    print(f'\n[exam_chroma_file] Examining chroma file: {os.path.basename(path_chroma)}')
    #[1] Create chroma if not exist.
    if not os.path.exists(path_chroma):
        text_warning = f'⚠️ [Warning] Chroma file not found at {path_chroma}. Creating chroma file from audio...'
        print(text_warning)
        NewSongRegistration.from_audio_to_chroma(path_audio, path_chroma, CHROMA_SETTINGS) 
    
    #[2] Load chroma and times from .npz file.
    data = np.load(path_chroma, allow_pickle=True)
    chroma, times = data['chroma'], data['times']

    #[3] Analyze chroma properties.
    file_size_kb = os.path.getsize(path_chroma) / 1024
    file_size_kb = os.path.getsize(path_chroma) / 1024

    #[4] Compile diagnostic info into a dict and print it. Add CHROMA_SETTINGS.
    dict_chroma_info = {
        'c_path_chroma':        path_chroma,
        'c_chroma_kB':          round(file_size_kb, 1),
        'c_chroma_shape':       chroma.shape,
    }
    dict_chroma_info.update(CHROMA_SETTINGS)
    
    #[5] Print the diagnostic info in a readable format.
    M1F.format_dict_beautifully(dict_chroma_info)
    return dict_chroma_info



def compare_two_dicts(path_or_dict1, path_or_dict2 = None):
    """
    Compare two dictionaries and print the differences in a readable format.
    If keys_to_compare is provided, only compare those keys; otherwise, compare all keys in dict1.
    """
    print(f'\n[compare_two_dicts] Comparing two dictionaries...')
    #[1] Load dicts from paths if given as file paths.
    dict1 = M1F.read_json_as_dict(path_or_dict1) if isinstance(path_or_dict1, str) else path_or_dict1
    if path_or_dict2 is not None:
        dict2 = M1F.read_json_as_dict(path_or_dict2) if isinstance(path_or_dict2, str) else path_or_dict2
    
    
    #[2] If not dict2, just show dict1 and ask user to confirm if it's correct.
    if path_or_dict2 is None:
        text_q =    f'\n[Review Dict] Please review the following dictionary and confirm if it is correct (yes/no):\n'\
                    f'{M1F.format_dict_beautifully(dict1)}'
        M1F.question_and_if_yes_action(text_q)
        
    #[3] Compare the two dicts and print differences.
    else:
        if dict1 == dict2:
            print('✅ The two dictionaries are identical.')
            return
        else:
            text_q =    f'⚠️ [Differences Found] The two dictionaries have differences. Please review the following comparison:\n'\
                        f'dict1:\n{M1F.format_dict_beautifully(dict1)}\n'\
                        f'dict2:\n{M1F.format_dict_beautifully(dict2)}\n'
            M1F.question_and_if_yes_action(text_q)



def convert_audio_to_mono(audio_path_or_array):
    """
    Convert audio to mono if it is stereo.
    """
    #[1] Check if the input is a file path or a numpy array
    if isinstance(audio_path_or_array, str):
        audio, sr = librosa.load(audio_path_or_array, sr=None, mono=False) 
    elif isinstance(audio_path_or_array, np.ndarray):
        audio = audio_path_or_array
        sr = 44100  # Default sampling rate, adjust if necessary
        
    #[2] If the audio has more than one channel, convert it to mono
    if audio.ndim > 1:
        print(f'-- Converting audio from {audio.shape[0]} channels to mono by averaging across channels.')
        audio_mono = librosa.to_mono(audio)
    else:
        #print(f'-- Audio is already mono with shape {audio.shape}. No conversion needed.')
        audio_mono = audio
    return audio_mono, sr


def read_audio_file_and_cvt_to_mono(path_audio, cvt_to_mono=True):
    """
    Read audio file. Convert to mono and save optionally.
    
    """
    #[1] Load audio with original sampling rate and channels.
    audio, sr = librosa.load(path_audio, sr=None, mono=False)
    
    #[3] Optionally convert to mono.
    if cvt_to_mono and audio.ndim > 1:
        print(f'-- Converting audio to mono and overwriting the original file: {os.path.basename(path_audio)}')
        audio, sr = convert_audio_to_mono(audio)
        #[4] Save (overwrite).
        sf.write(path_audio, audio, sr)

    return audio, sr






def compute_stft_spectrogram(audio_mono, sr):
    """
    Compute magnitude spectrogram using STFT.
    Returns:
        S (np.ndarray): Magnitude spectrogram
        times (np.ndarray): Frame time axis in seconds
    """
    #[1] Compute complex STFT
    S_complex = librosa.stft(
        y=audio_mono,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_type,
        center=True)
    
    #[2] Take magnitude
    S = np.abs(S_complex)
    
    #[3] Convert to dB scale for better dynamic range (optional)
    times = librosa.frames_to_time(
        np.arange(S.shape[1]),
        sr=sr,
        hop_length=hop_length)
    
    return S, times






def add_path_to_audio(  work_or_recd_meta, 
                        kn_to_fine  = 'file name', 
                        kn_to_add   = 'path original audio', 
                        audio_ext   = 'mp3'):
    """
    Used in work_meta: to add "path original audio" based on "file name".
    Used in recd_meta: to add "path recording audio" based on "file name".
    """
    #[1] Make sure we can get "file name" from work_meta, which is the base name of the original audio file without extension.
    _file_name = work_or_recd_meta.get(kn_to_fine, "")
    if _file_name in ["", None]:
        raise ValueError(f'\n❌ [Error] Key "{kn_to_fine}" is missing in the work_meta. Cannot determine the path of original audio file without it.')
    
    #[2] Construct the path of original audio file.
    path_original_audio = f'{os.path.join(DIR_SONG_BANK, _file_name)}.{audio_ext}' if work_or_recd_meta.get(kn_to_fine, "") else None
    work_or_recd_meta[kn_to_add] = path_original_audio if os.path.exists(path_original_audio) else None
    #[3] Verify audio file existance and report warning if not exist.
    if not os.path.exists(path_original_audio):
        raise ValueError(f'\n❌ [Error] Original audio file not found, please check:\n{path_original_audio}')
    return work_or_recd_meta



def dict_key_check_with_template(dict_to_check, dict_template):
    
    """
    Template as master. Check if all keys in the template exist in the dict to check.
    Find "track_ID" if available; otherwise, find "work_ID".
    """
    #[1] Find ID for reporting.
    show_ID = None
    if 'track_ID' in dict_to_check:
        show_ID = dict_to_check['track_ID']
    elif 'work_ID' in dict_to_check:
        show_ID = dict_to_check['work_ID']
    print(f'\n[dict_key_check_with_template] {show_ID}')
    
    #[2] Check if all keys in the template exist in the dict to check.
    all_keys_correct = True
    for key in dict_template.keys():
        if key not in dict_to_check:
            print(f'⚠️ [Warning] Key "{key}" exists in template but is missing in the dictionary to check.')
            all_keys_correct = False
    #[8] Final output.
    if all_keys_correct:
        return True
    else:
        return False



"""
## 2. Directory Architecture (PoC Stage)

reference/
├── song_001/
│   ├── song_001_meta.json
│   ├── version_01/
│   │   ├── version_01.mp3
│   │   ├── version_01.npz
│   │   └── version_01.json
│   └── version_02/
│       ├── version_02.mp3
│       ├── version_02.npz
│       └── version_02.json
├── song_002/
│   └── ...



"""


# ============================================================
# [3] Utility Function - End
# ============================================================





# ============================================================
# [4] Reference Check - Start
# ============================================================


def df_convert_bool_to_symbol(df, true_symbol=None, false_symbol=None):
    if true_symbol is not None:
        df = df.replace({True: true_symbol})
    if false_symbol is not None:
        df = df.replace({False: false_symbol})
    return df



class Full_work_ID_Check:
    """
    Purpose: Check if every unique work_ID has a corresponding reference folder with complete work_meta.
    """
    def __init__(self):
        
        #[1] Settings.
        list_col_print = ['track_ID', 'Canonical Title', 'Original Artist', 'Recording Artist']
        
        #[2] Initialize instance variables.
        self.need_update_work_meta      = False
        self.list_work_ID_need_update   = []
        self.agreed_to_update           = False
        self.list_work_ID_processed     = []
        
        #[4] Read track table and print selected columns for quick review.
        self.df_track_list, self.df_track_print = read_track_table(list_col_print)
        
        #[5] Verify work-level content for all work_IDs.
        self.varify_work_level()
        
        #[6] If there are work_IDs that need update, ask user if they want to update now.
        if len(self.list_work_ID_need_update) > 0:
            text_q2 =   f'\n⚠️ [Warning] The following work_IDs need update:\n{self.list_work_ID_need_update}\n'\
                        f'Do you want to update them now? (yes/no)'
            if M1F.question_and_if_yes_action(text_q2):
                self.agreed_to_update = True
                for i_work_ID in self.list_work_ID_need_update:
                    path_work_meta = os.path.join(DIR_REF, i_work_ID, f'{i_work_ID}_work meta.json')
                    NewSongRegistration.create_work_meta(   work_ID     = i_work_ID, 
                                                            df_track_list = self.df_track_list,
                                                            path_work_meta = path_work_meta,
                                                            auto_save   = True)
                    self.list_work_ID_processed.append(i_work_ID)
                    #break  # For testing, just do the first one. Remove this line to process all.
            
        #[8] After update, re-verify work-level content to confirm all are correct now.
        self.varify_work_level()
        if len(self.list_work_ID_need_update) == 0:
            print(f'\n✅ All work_IDs are verified and correct now.')
        else:
            print(f'-- list_work_ID_processed: {self.list_work_ID_processed}')
            print(f'-- The following work_IDs still need update:\n{self.list_work_ID_need_update}')
            print(f'-- If this keep happening, check WORK_META_TEMPLATE and LIST_COL_WORK_META.')
        


    def varify_work_level(self):
        print(f'\n[verify_work_level]')
        #[1] Generate unique list of work_IDs from df_track_list.
        list_work_ID = self.df_track_list['track_ID'].apply(lambda x: x.split('-')[0]).unique()
        print(f'-- Unique work_IDs found in track table {len(list_work_ID)}:\n{list_work_ID}')

        #[2] list_work_folders = list of folder names in DIR_REF.
        list_work_folders = [d for d in os.listdir(DIR_REF) if os.path.isdir(os.path.join(DIR_REF, d))]
        print(f'-- Work folders found in reference directory {len(list_work_folders)}:\n{list_work_folders}')
        
        #[3] Examine each work_ID and check if corresponding folder and work_meta exist and correct.
        df_work_check = pd.DataFrame(columns=['work_ID', 'work_folder_exist', 'work_meta_exist', 'work_meta_correct'])
        for i_work_ID in list_work_ID:
            path_work_meta = os.path.join(DIR_REF, i_work_ID, f'{i_work_ID}_work meta.json')
            work_folder_exist       = i_work_ID in list_work_folders
            work_meta_exist         = os.path.exists(path_work_meta)
            work_meta_correct       = self.assert_work_meta_integrity(path_work_meta, i_work_ID, False) if work_meta_exist else False
            work_meta_verified      = work_folder_exist and work_meta_exist and work_meta_correct  
            #[4] Append the check result for this work_ID to df_work_check.
            df_work_check = df_work_check._append({
                'work_ID':              i_work_ID,
                'work_folder_exist':    work_folder_exist,
                'work_meta_exist':      work_meta_exist,
                'work_meta_correct':    work_meta_correct,
                'verified':             work_meta_verified
            }, ignore_index=True)
            
            #[5] If anything is missing or incorrect, set self.need_update_work_meta = True to trigger update later.
            if not work_meta_verified:
                self.need_update_work_meta = True
            
        print(f'\n[verify_work_level] Summary of work-level check:\n'\
                f'{tabulate(df_convert_bool_to_symbol(df_work_check, "✅"), headers="keys", tablefmt="orgtbl", showindex=False)}\n')
        #[6] Collect work_IDs that need update for later processing.
        self.list_work_ID_need_update = df_work_check[df_work_check['verified'] == False]['work_ID'].tolist()

        return df_work_check



    def assert_work_meta_integrity(self, path_work_meta, work_ID_expected, step_print=True):
        """
        Check if all the desired content in the generated work_meta dict exist and correct.    
        """
        
        #[1] File existance check.
        if not os.path.exists(path_work_meta):
            print(f'⚠️ [Warning] Work meta file not found at {path_work_meta}.')
            return False
        print(f'✓ Work meta file exists.') if step_print else None
        
        #[2] Check file name format.
        bsn_work_meta = os.path.basename(path_work_meta)
        if not bsn_work_meta.endswith('_work meta.json'):
            print(f'⚠️ [Warning] Work meta file name "{bsn_work_meta}" does not follow the expected format "<work_ID>_work meta.json".')
            return False
        print(f'✓ Work meta file name format is correct.') if step_print else None
        
        #[3] Check if work_ID in file name matches the "work_ID" field in the dict.
        work_ID_from_bsn    = bsn_work_meta.split('_work meta.json')[0]
        if work_ID_from_bsn != work_ID_expected:
            print(f'⚠️ [Warning] Work ID in file name "{work_ID_from_bsn}" does not match the "work_ID" field in the dict "{work_ID_expected.get("work_ID", "")}".')
            return False
        print(f'✓ Work ID in file name matches the "work_ID" field in the dict.') if step_print else None
        
        #[5] Check if all desired keys in WORK_META_TEMPLATE exist in the dict.
        dict_work_meta = M1F.read_json_as_dict(path_work_meta)
        if not dict_key_check_with_template(dict_work_meta, WORK_META_TEMPLATE):
            return False
        print(f'✓ All desired content in the work_meta dict exist and correct.') if step_print else None
        
        #[6] Check if every keys has a value (not "" or None).
        for key in dict_work_meta:
            if key not in dict_work_meta:
                print(f'⚠️ [Warning] Key "{key}" is missing in the work_meta dict.')
                return False
            elif dict_work_meta[key] in ["", None]:
                print(f'⚠️ [Warning] Key "{key}" has an empty value in the work_meta dict.')
                return False
        print(f'✓ Every key in the work_meta dict has a value (not "" or None).') if step_print else None
        
        #[8] Check if keys that should be values are not stored as strings.
        M1F.dict_covert_str_numbers_to_actual_numbers(dict_work_meta, keys_should_be_number, path_work_meta)
        print(f'✓ Keys that should be values are stored in correct data types.') if step_print else None
        
        return True
        





class Full_track_ID_Check:
    """
    Purpose: Check if every track_ID has a corresponding recording-level folder with complete recd_meta and chroma file.
    """
    def __init__(self):
        #[1] Settings.
        list_col_print = ['track_ID', 'Canonical Title', 'Original Artist', 'Recording Artist', 'Release Year']
        
        #[2] Initialize instance variables.
        self.need_update_recd_meta      = False
        self.list_track_ID_need_update   = []
        self.agreed_to_update           = False
        self.list_track_ID_processed     = []

        #[4] Read track table and print selected columns for quick review.
        self.df_track_list, self.df_track_print = read_track_table(list_col_print)
        
        #[5] Verify work-level content for all work_IDs.
        self.varify_recording_level()

        #[6] If there are track_IDs that need update, ask user if they want to update now.
        if len(self.list_track_ID_need_update) > 0:
            text_q3 =   f'\n⚠️ [Warning] The following track_IDs need update:\n{self.list_track_ID_need_update}\n'\
                        f'Do you want to update them now? (yes/no)'
            if M1F.question_and_if_yes_action(text_q3):
                self.agreed_to_update = True
                for i_track_ID in self.list_track_ID_need_update:
                    print(f'\n🔄 Updating track_ID {i_track_ID}...')
                    i_work_ID, i_recd_ID = i_track_ID.split('-')
                    path_recd_meta  = os.path.join(DIR_REF, i_work_ID, i_recd_ID, f'{i_recd_ID}_recording meta.json')
                    path_chroma     = os.path.join(DIR_REF, i_work_ID, i_recd_ID, f'{i_recd_ID}_chroma.npz')
                    NewSongRegistration.create_recd_meta_and_chroma(    track_ID        = i_track_ID,
                                                                        df_track_list   = self.df_track_list,
                                                                        path_recd_meta  = path_recd_meta,
                                                                        path_chroma     = path_chroma,
                                                                        auto_save       = True)
                    self.list_track_ID_processed.append(i_track_ID)
                    break  # For testing, just do the first one. Remove this line to process all.
                
        #[8] After update, re-verify recording-level content to confirm all are correct now.
        self.varify_recording_level()
        if len(self.list_track_ID_need_update) == 0:
            print(f'\n✅ All track_IDs are verified and correct now.')
        else:
            print(f'-- list_track_ID_processed: {self.list_track_ID_processed}')
            print(f'-- The following track_IDs still need update:\n{self.list_track_ID_need_update}')
            print(f'-- If this keep happening, check RECD_META_TEMPLATE and LIST_COL_RECD_META.')





    def varify_recording_level(self):
        print(f'\n[varify_recording_level]')
        #[1] Generate unique list of recording_IDs from df_track_list.
        list_track_ID = self.df_track_list['track_ID'].unique()
        print(f'-- Unique track_ID found in track table {len(list_track_ID)}:\n{list_track_ID}')
        
        #[3] Examine each work_ID and check if corresponding folder and work_meta exist and correct.
        df_recd_check = pd.DataFrame(columns=['track_ID', 'recd_folder_exist', 'recd_exist', 'recd_OK', 'chroma_exist', 'chroma_OK'])
        for i_track_ID in list_track_ID:
            i_work_ID, i_recd_ID = i_track_ID.split('-')
            path_recd_meta          = os.path.join(DIR_REF, i_work_ID, i_recd_ID, f'{i_recd_ID}_recording meta.json')
            path_chroma = os.path.join(DIR_REF, i_work_ID, i_recd_ID, f'{i_recd_ID}_chroma.npz')
            recd_folder_exist       = os.path.exists(os.path.join(DIR_REF, i_work_ID, i_recd_ID))
            recd_exist              = os.path.exists(os.path.join(DIR_REF, i_work_ID, i_recd_ID, f'{i_recd_ID}_recording meta.json'))
            recd_OK                 = self.assert_recd_meta_integrity(path_recd_meta, i_recd_ID, False) if recd_exist else False
            chroma_exist            = os.path.exists(path_chroma)
            chroma_OK               = self.assert_chroma_file_integrity(path_chroma, path_recd_meta, i_track_ID) if chroma_exist else False
            recording_verified      = recd_folder_exist and recd_exist \
                                        and recd_OK and chroma_exist \
                                        and chroma_OK 
            
            #[4] Append the check result for this recording_ID to df_recd_check.
            df_recd_check = df_recd_check._append({
                'track_ID':             i_track_ID,
                'recd_folder_exist':    recd_folder_exist,
                'recd_exist':           recd_exist,
                'recd_OK':              recd_OK,
                'chroma_exist':         chroma_exist,
                'chroma_OK':            chroma_OK,
                'verified':             recording_verified
            }, ignore_index=True)
            
            #[5] If anything is missing or incorrect, set self.need_update_recd_meta = True to trigger update later.
            if not recording_verified:
                self.need_update_recd_meta = True
        print(f'\n[verify_recording_level] Summary of recording-level check:\n'\
                f'{tabulate(df_convert_bool_to_symbol(df_recd_check, "✅"), headers="keys", tablefmt="orgtbl", showindex=False)}\n')
        #[6] Collect recording_IDs that need update for later processing.
        self.list_track_ID_need_update = df_recd_check[df_recd_check['verified'] == False]['track_ID'].tolist()



    def assert_recd_meta_integrity(self, path_recd_meta, recording_ID_expected, step_print=True):
        """
        Check if all the desired content in the generated recd_meta dict exist and correct.    
        """

        # [1] File existence check.
        if not os.path.exists(path_recd_meta):
            print(f'⚠️ [Warning] Recording meta file not found at {path_recd_meta}.')
            return False
        if step_print:
            print(f'✓ Recording meta file exists.')

        # [2] Check file name format.
        bsn_recd_meta = os.path.basename(path_recd_meta)
        if not bsn_recd_meta.endswith('_recording meta.json'):
            print(f'⚠️ [Warning] Recording meta file name "{bsn_recd_meta}" does not follow the expected format "<recording_ID>_recording meta.json".')
            return False
        if step_print:
            print(f'✓ Recording meta file name format is correct.')

        # [3] Check if recording_ID in file name matches the "recording_ID" field in the dict.
        recording_ID_from_bsn = bsn_recd_meta.split('_recording meta.json')[0]
        # Read dict
        dict_recd_meta = M1F.read_json_as_dict(path_recd_meta)
        recording_ID_from_dict = dict_recd_meta.get('recording_ID', None)
        if recording_ID_from_bsn != str(recording_ID_expected):
            print(f'⚠️ [Warning] Recording ID in file name "{recording_ID_from_bsn}" does not match the expected recording_ID "{recording_ID_expected}".')
            return False
        if step_print:
            print(f'✓ Recording ID in file name matches the expected recording_ID.')

        # [4] Check if all desired keys in RECD_META_TEMPLATE exist in the dict.
        if not dict_key_check_with_template(dict_recd_meta, RECD_META_TEMPLATE):
            return False
        if step_print:
            print(f'✓ All desired content in the recd_meta dict exist and correct.')

        # [5] Check if every key has a value (not "" or None).
        for key in RECD_META_TEMPLATE:
            if key not in dict_recd_meta:
                print(f'⚠️ [Warning] Key "{key}" is missing in the recd_meta dict.')
                return False
            elif dict_recd_meta[key] in ["", None]:
                print(f'⚠️ [Warning] Key "{key}" has an empty value in the recd_meta dict.')
                return False
        if step_print:
            print(f'✓ Every key in the recd_meta dict has a value (not "" or None).')

        # [6] Check if keys that should be numbers are stored in correct data types.
        M1F.dict_covert_str_numbers_to_actual_numbers(dict_recd_meta, keys_should_be_number, path_recd_meta)
        if step_print:
            print(f'✓ Keys that should be values are stored in correct data types.')
            
        #[7] Mathematical relationship check.
        if not self.recd_meta_math_relationship_check(dict_recd_meta):
            print(f'⚠️ [Warning] Mathematical relationship check failed for recd_meta.')
            return False

        return True
    
    def recd_meta_math_relationship_check(self, dict_recd_meta):
        """
        Check if the mathematical relationship between audio properties holds in recd_meta.
        """
        #[1] a_n_samples = a_sampling_rate * a_duration_sec
        a_n_samples         = dict_recd_meta.get('a_n_samples', None)
        a_sampling_rate     = dict_recd_meta.get('a_sampling_rate', None)
        a_duration_sec      = dict_recd_meta.get('a_duration_sec', None)
        if not np.isclose(a_n_samples, a_sampling_rate * a_duration_sec, rtol=0.01):
            text_warning =  f'⚠️ [Warning] The relationship a_n_samples ≈ a_sampling_rate * a_duration_sec does not hold in recd_meta.\n'\
                            f'Got a_n_samples={a_n_samples}, a_sampling_rate={a_sampling_rate}, a_duration_sec={a_duration_sec}.'   
            print(text_warning)
            return False
        
        #[2] c_chroma_shape[1] = a_n_samples // hop_length + 1 (if center=True in STFT)
        c_chroma_shape  = dict_recd_meta.get('c_chroma_shape', None)
        hop_length      = dict_recd_meta.get('hop_length', None)
        expected_frames = a_n_samples // hop_length + 1 if c_chroma_shape is not None else None
        if c_chroma_shape[1] != expected_frames:
            text_warning =  f'⚠️ [Warning] The relationship c_chroma_shape[1] ≈ a_n_samples // hop_length + 1 does not hold in recd_meta.\n'\
                            f'Got c_chroma_shape={c_chroma_shape}, a_n_samples={a_n_samples}, expected_frames={expected_frames}.'
            print(text_warning)
            return False
        
        return True
    
    
    
    

    @staticmethod
    def assert_chroma_file_integrity(path_chroma, path_recd_meta, track_ID):
        """
        Read chroma and make sure it matches the c_chroma_shape in the recd_meta.   
        """
        print(f'\n[assert_chroma_file_integrity] {track_ID}')
        #[1] Make sure two files exist.
        if not os.path.exists(path_chroma):
            print(f'⚠️ [Warning] Chroma file not found at {path_chroma}.')
            return False
        if not os.path.exists(path_recd_meta):
            print(f'⚠️ [Warning] Recording meta file not found at {path_recd_meta}..')
            return False        
        
        #[3] Load chroma and times from .npz file.
        data = np.load(path_chroma, allow_pickle=True)
        chroma, times = data['chroma'], data['times']
        chroma_shape = chroma.shape
        
        #[4] Load recd_meta and get the expected chroma shape.
        dict_recd_meta = M1F.read_json_as_dict(path_recd_meta)
        expected_chroma_shape = dict_recd_meta.get('c_chroma_shape', None)
        if expected_chroma_shape is None:
            print(f'⚠️ [Warning] "c_chroma_shape" is missing in the recd_meta dict. Cannot verify chroma file integrity without it.')
            return False
        
        #[5] Compare the actual chroma shape with the expected chroma shape.
        if chroma_shape != tuple(expected_chroma_shape):
            print(f'⚠️ [Warning] Chroma shape {chroma_shape} does not match the expected chroma shape {expected_chroma_shape} in recd_meta.')
            return False
        
        return True





# ============================================================
# [4] Reference Check - End
# ============================================================












# ============================================================
# [6] Registration Class - Start
# ============================================================


class NewSongRegistration:
    """
    Handles registration and metadata management for new songs in the LyraSense system.
    Creates work-level and recording-level directories and metadata files.
    """
    
    
    def __init__(self, dir_REF, track_ID):
        
        #[1] Register instance variables.
        self.dir_REF        = dir_REF
        self.work_meta      = None
        self.recd_meta = None
        
        
        #[2] Read track catalog.
        self.df_track_list  = read_track_table()
        
        
        #[4] Work-level.
        self.track_ID       = track_ID
        assert len(self.track_ID.split('-')) == 2, f'Error: track_ID "{self.track_ID}" should be in the format "workID-recordingID".'
        self.work_ID, self.recording_ID     = self.track_ID.split('-')
        self.dir_work       = os.path.join(self.dir_REF, self.work_ID)
        self.path_work_meta = os.path.join(self.dir_work, f'{self.work_ID}_work meta.json')
        
        #[5] Recording-level.
        self.dir_recording  = os.path.join(self.dir_work, self.recording_ID)
        self.path_chroma    = os.path.join(self.dir_recording, f'{self.recording_ID}_chroma.npz')
        self.path_recd_meta = os.path.join(self.dir_recording, f'{self.recording_ID}_recording meta.json')
        self.path_audio_file = None

        self.verify_content()




    def verify_content(self):
        #[1] Verify work-level and recording-level directories and meta files.
        self.verify_dir_work_level()
        sys.exit('Checking content at recording level ...')
        self.verify_dir_recording_level()
        
        
        
        
        self.verify_audio_file()
        self.verify_chroma_file()



    
    def verify_dir_work_level(self):
        #[1] Check if work directory exists. Create if not.
        if not os.path.exists(self.dir_work):
            os.makedirs(self.dir_work)
            print(f'☑️ Created new work directory: {self.dir_work}')
        else:
            print(f'✅ Work directory already exists: {self.dir_work}')
        need_update_meta = False

        #[3] Read self.path_work_meta to check if it's empty.
        if os.path.exists(self.path_work_meta):
            work_meta = M1F.read_json_as_dict(self.path_work_meta)
            if not work_meta:
                print(f'⚠️ [Warning] Work meta file is empty.')
                need_update_meta = True
            else:
                print(f'✅ Work meta file already exists and is not empty.')
        else:
            print(f'⚠️ [Warning] Work meta file not found.')
            need_update_meta = True
        
        #[4] If meta file is empty or not exist.
        #print(f'-- Forcing need_update_meta = True for testing ...')
        #need_update_meta = True
        if need_update_meta:
            work_meta = self.create_work_meta(auto_save=True)
        self.work_meta = work_meta

    @staticmethod
    def create_work_meta(   work_ID,
                            df_track_list,
                            path_work_meta,
                            auto_save=True):
        """
        This "create_work_meta" function should:
        1. Use track_ID to find the corresponding row in df_track_list.
        2. Extract relevant metadata fields from that row.
        3. Fill in the WORK_META_TEMPLATE with the extracted metadata.
        4. Save dict as json to self.path_work_meta. 
        """
        
        print(f'\n[create_work_meta] {work_ID}')
        #[1] Create work-level directory if not exist.
        dir_work = os.path.dirname(path_work_meta)
        if not os.path.exists(dir_work):
            os.makedirs(dir_work)
            print(f'☑️ Created new work directory for saving work meta: {dir_work}')
        
        #[3] Create a work meta file. Quickly fill with easy-to-get info.
        work_meta = WORK_META_TEMPLATE.copy()
        work_meta['work_ID'] = work_ID        
        
        #[4] Look up the row in df_track_list (from excel) matching this work_ID.
        work_meta_found = lookup_df_track_list_by_work_ID(work_meta['work_ID'], df_track_list, LIST_COL_WORK_META)        
        work_meta = M1F.add_lookedup_values_to_dict_template(work_meta, work_meta_found)
        work_meta = add_path_to_audio(work_meta, 'file name', 'path original audio')
        work_meta = {k: work_meta.get(k, None) for k in WORK_META_TEMPLATE.keys()}  # Reorder.

        #[8] Compare with template (only keys, no value) and report missing keys.
        M1F.compare_dict_with_template(work_meta, WORK_META_TEMPLATE)

        #[7] Save dict as json to self.path_work_meta.
        M1F.format_dict_beautifully(work_meta, keys_should_be_number)
        if auto_save:
            M1F.save_dict_as_json(work_meta, path_work_meta, True)
            print(f'-- ☑️ Created work meta file at {path_work_meta}.')
        else:
            print(f'-- ⚠️ [Warning] auto_save is disabled. Please review and save mannually.')
        return work_meta






    def verify_dir_recording_level(self):
        #[1] Start with the assumption that we need to update the recording meta.
        need_update_meta = True
        
        #[1] Check if recording directory exists. Create if not.
        if not os.path.exists(self.dir_recording):
            os.makedirs(self.dir_recording)
            print(f'☑️ Created new recording directory: {self.dir_recording}')
            need_update_meta = True
        else:
            print(f'✅ Recording directory already exists: {self.dir_recording}')
            need_update_meta = False
        
        #[3] Check if self.path_recd_meta exists. Create if not.
        if not os.path.exists(self.path_recd_meta):
            print(f'⚠️ [Warning] Recording meta file not found. Creating a new ones.')
            need_update_meta = True
        else:
            print(f'✅ Recording meta file already exists: {self.path_recd_meta}')
        
        #[4] If meta file is empty or not exist, create a new one with default values.
        if need_update_meta:
            meta_recording = self.create_recd_meta(auto_save=True)
        self.recd_meta = meta_recording
    

    
    @staticmethod
    def create_recd_meta_and_chroma(track_ID, 
                                    df_track_list, 
                                    path_recd_meta, 
                                    path_chroma, 
                                    auto_save=True):
        """
        This "create_recd_meta_and_chroma" function should:
        1. Use track_ID to find the corresponding row in df_track_list.
        2. Extract relevant metadata fields from that row.
        3. Fill in the RECD_META_TEMPLATE with the extracted metadata.
        4. Read the audio file and extract audio properties (e.g., duration, sampling rate) to fill in the meta.
        5. Compute chroma feature and fill in related meta fields (e.g., chroma shape, chroma times).
        6. Save dict as json to self.path_recd_meta.
        """
        
        print(f'\n[create_recd_meta_and_chroma] {track_ID}')
        #[1] Create recording-level directory if not exist.
        dir_recording = os.path.dirname(path_recd_meta)
        if not os.path.exists(dir_recording):
            os.makedirs(dir_recording)
            print(f'☑️ Created new recording directory: {dir_recording}')
             
        #[3] Create a recording meta, quickly fill with easy-to-get info.
        recd_meta = RECD_META_TEMPLATE.copy()
        recd_meta['track_ID'] = track_ID
        recd_meta['work_ID'], recd_meta['recording_ID'] = track_ID.split('-')   
        
        #[4] Look up the row in df_track_list (from excel) matching this track_ID.
        recd_meta_found = lookup_df_track_list_by_track_ID(recd_meta['track_ID'], df_track_list, LIST_COL_RECD_META)
        recd_meta = M1F.add_lookedup_values_to_dict_template(recd_meta, recd_meta_found)
        recd_meta = add_path_to_audio(recd_meta, 'file name', 'path recording audio')
        recd_meta = {k: recd_meta.get(k, None) for k in RECD_META_TEMPLATE.keys()}  # Reorder.

        #[5] Read info from audio file.
        path_audio = recd_meta.get('path recording audio', None)
        a_meta = exam_audio_file(path_audio)
        recd_meta = M1F.add_lookedup_values_to_dict_template(recd_meta, a_meta)
        
        #[6] Create chroma feature and times arrays.
        NewSongRegistration.from_audio_to_chroma(path_audio, path_chroma, CHROMA_SETTINGS)
        c_meta = exam_chroma_file(path_chroma, path_audio)
        recd_meta = M1F.add_lookedup_values_to_dict_template(recd_meta, c_meta)
        
        #[8] Compare with template (only keys, no value) and report missing keys.
        M1F.compare_dict_with_template(recd_meta, RECD_META_TEMPLATE)

        #[7] Save dict as json to self.path_work_meta.
        M1F.format_dict_beautifully(recd_meta, keys_should_be_number)
        if auto_save:
            M1F.save_dict_as_json(recd_meta, path_recd_meta, True)
            print(f'-- ☑️ Created recording meta file at {path_recd_meta}.')
        else:
            print(f'-- ⚠️ [Warning] auto_save is disabled. Please review and save mannually.')
        return recd_meta


    









    def verify_audio_file(self):
        """
        Under recording-level directory, check if there is exactly one audio file.
        """
        print(f'\n[verify_audio_file]')
        #[1] List audio files in recording directory.
        audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg']
        audio_files = [f for f in os.listdir(self.dir_recording) if os.path.splitext(f)[1].lower() in audio_extensions]
        dir_name = os.path.basename(self.dir_recording)
        
        #[2] Check if there is exactly one audio file.
        if len(audio_files) == 0:
            print(f'⚠️ [Warning] No audio files found in {self.dir_recording}. Let me see if I can copy from the song bank ...')
            
            # Get "File Name" from self.recd_meta and try to find it in the df_track_list to copy the audio file from the song bank.
            file_name = self.recd_meta.get('file name', None)
            path_audio_from_bank = f'{os.path.join(DIR_SONG_BANK, file_name)}.mp3' if file_name else None
            if file_name and os.path.exists(path_audio_from_bank):
                M1F.copy_file(path_audio_from_bank, os.path.join(self.dir_recording, file_name))
                print(f'☑️ Copied audio file from song bank: {path_audio_from_bank} to {self.dir_recording}')
                return self.verify_audio_file()  # Re-run verification after copying
            else:
                print(f'⚠️ [Warning] Cannot find audio file at "{path_audio_from_bank}"\nPlease check the "File Name" field in the recording meta or add the audio file manually.')
                return False
        elif len(audio_files) > 1:
            print(f'⚠️ [Warning] Multiple audio files found in {self.dir_recording}: {audio_files}. Please keep only one audio file.')
            return False
        else:
            print(f'✅ Found exactly one audio file in {dir_name}: {audio_files[0]}')
            self.path_audio_file = os.path.join(self.dir_recording, audio_files[0])
            
            #[5] Exam the audio file for diagnostics.
            audio_meta = {}
            audio_meta['a_path_audio_file'] = self.path_audio_file
            audio_meta.update(exam_audio_file(self.path_audio_file))
            
            #[6] Update recording meta with audio file info and save.
            if self.recd_meta is not None:
                self.recd_meta.update(audio_meta)
                M1F.save_dict_as_json(self.recd_meta, self.path_recd_meta, True)
                print(f'☑️ Updated recording meta with audio file info at {self.path_recd_meta}.')
            
            return True
        
        
        

    def validate_chroma_file(self):
        """
        Check whether self.path_chroma exists and contains valid 'chroma' and 'times' arrays.
        Returns True if valid, False otherwise.
        """
        print(f'\n[validate_chroma_file]')
        if not os.path.exists(self.path_chroma):
            return False
        try:
            #[2] Load chroma file and check for required keys and shapes.
            data = np.load(self.path_chroma, allow_pickle=True)
            if 'chroma' not in data or 'times' not in data:
                print(f'⚠️ [Warning] {self.path_chroma} missing required keys (chroma, times).')
                return False
            chroma, times = data['chroma'], data['times']
            if chroma.ndim != 2 or chroma.shape[0] != n_chroma or chroma.shape[1] != times.shape[0]:
                print(f'⚠️ [Warning] Chroma file shape mismatch: chroma.shape={chroma.shape}, times.shape={times.shape}')
                return False
            else:
                print(f'✅ Valid chroma file found at {self.path_chroma}.')
                
                #[5] Exam the chroma file for diagnostics.
                chroma_meta = {}
                chroma_meta['c_path_chroma'] = self.path_chroma
                chroma_meta.update(exam_chroma_file(self.path_chroma))
                
                #[6] Update recording meta with chroma file info and save.
                if self.recd_meta is not None:
                    self.recd_meta.update(chroma_meta)
                    M1F.save_dict_as_json(self.recd_meta, self.path_recd_meta, True)
                    print(f'☑️ Updated recording meta with chroma file info at {self.path_recd_meta}.')
                return True
        except Exception as e:
            print(f'⚠️ [Warning] Failed to load chroma file: {e}')
            return False


    def verify_chroma_file(self):
        """
        Check if the chroma feature file exists and is valid. If not, create the chroma feature file.
        """
        print(f'\n[verify_chroma_file]')
        #[1] Guard: cannot create chroma without an audio file.
        if not self.path_audio_file:
            print(f'⚠️ [Warning] No audio file available. Skipping chroma verification.')
            return False
        
        #[2] Check if chroma feature file exists and is valid. If not, create the chroma feature file.
        if self.validate_chroma_file():
            return True
        
        print(f'-- Creating chroma feature file for {self.path_audio_file} ...')
        try:
            self.from_audio_to_chroma(self.path_audio_file)
        except Exception as e:
            print(f'❌ Failed to create chroma file: {e}')
            return False
        if os.path.exists(self.path_chroma, self.path_chroma):
            print(f'☑️ Chroma file created: {self.path_chroma}')
            return True
        print(f'❌ Chroma file was not created at {self.path_chroma}.')
        return False

    
    @staticmethod
    def from_audio_to_chroma(audio_path, path_chroma, dict_chroma_settings):
        #[1] Load audio, convert to mono, and resample if needed.
        audio_mono, sr = convert_audio_to_mono(audio_path)
        if resample_audio and sr != expect_sr:
            text_warning =  f'⚠️ [Warning] The sampling rate of the audio file ({sr} Hz) does not match the expected sampling rate for chroma computation ({expect_sr} Hz).\n'\
                            f'Resampling audio from {sr} Hz to {expect_sr} Hz for chroma computation.'
            print(text_warning)
            audio_mono = librosa.resample(audio_mono, orig_sr=sr, target_sr=expect_sr)
            sr = expect_sr
        
        #[2] Compute chroma feature.
        chroma, times = compute_chroma_feature(audio_mono, sr, dict_chroma_settings)
        
        #[3] Save chroma feature as .npz file.
        np.savez(path_chroma, chroma=chroma, times=times)
        print(f'☑️ Chroma feature computed and saved to {path_chroma}.')
    
    



def compute_chroma_feature(audio_mono, sr, dict_chroma_settings):
    """
    Compute chroma feature for DTW-based alignment.
    """
    # [1] Read settings from dict_chroma_settings with defaults and warn if missing
    required_keys = [
        'hop_length', 'n_chroma', 'bins_per_octave', 'fmin', 'n_octaves',
        'feature_type', 'normalize_feature', 'apply_log_compression',
        'apply_smoothing', 'smoothing_window', 'apply_cens_quantization',
        'energy_threshold'
    ]
    missing_keys = [k for k in required_keys if k not in dict_chroma_settings]
    if missing_keys:
        print(f"⚠️ [Warning] Missing chroma settings: {missing_keys}")
        
    # [2] Compute chroma feature
    if feature_type == 'chroma_cqt':
        chroma = librosa.feature.chroma_cqt(y               = audio_mono,
                                            sr              = sr,
                                            hop_length      = dict_chroma_settings['hop_length'],
                                            n_chroma        = dict_chroma_settings['n_chroma'],
                                            bins_per_octave = dict_chroma_settings['bins_per_octave'],
                                            fmin            = dict_chroma_settings['fmin'],
                                            n_octaves       = dict_chroma_settings['n_octaves'])
    elif feature_type == 'chroma_stft':
        chroma = librosa.feature.chroma_stft(y              = audio_mono,
                                            sr              = sr,
                                            hop_length      = dict_chroma_settings['hop_length'],
                                            n_chroma        = dict_chroma_settings['n_chroma'],
                                            fmin            = dict_chroma_settings['fmin'])
    elif feature_type == 'chroma_cens':
        chroma = librosa.feature.chroma_cens(y              = audio_mono,
                                            sr              = sr,
                                            hop_length      = dict_chroma_settings['hop_length'],
                                            n_chroma        = dict_chroma_settings['n_chroma'],
                                            bins_per_octave = dict_chroma_settings['bins_per_octave'],
                                            fmin            = dict_chroma_settings['fmin'])
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    # [3] Post-processing
    if dict_chroma_settings['apply_log_compression']:
        chroma = np.log1p(chroma)
    if dict_chroma_settings['apply_smoothing']:
        from scipy.ndimage import uniform_filter1d
        chroma = uniform_filter1d(chroma, 
                                  size=dict_chroma_settings['smoothing_window'], 
                                  axis=1, 
                                  mode='nearest')
    if dict_chroma_settings['apply_cens_quantization']:
        # Quantize to CENS (Chroma Energy Normalized Statistics)
        chroma = np.round(chroma * 100) / 100.0

    # [4] Silence masking
    if dict_chroma_settings['energy_threshold'] is not None:
        energy = np.sum(np.abs(audio_mono)) / len(audio_mono)
        if energy < dict_chroma_settings['energy_threshold']:
            print(f"⚠️ [Warning] Audio energy below threshold: {energy:.5f} < {dict_chroma_settings['energy_threshold']}. This may affect chroma quality.")

    # [5] Optionally normalize each frame to unit norm for DTW stability
    if normalize_feature:
        chroma = librosa.util.normalize(chroma, axis=0)

    # [6] Compute time axis for frames
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, 
                                   hop_length=dict_chroma_settings['hop_length'])

    return chroma, times









# ============================================================
# [6] Registration Class - End
# ============================================================
















