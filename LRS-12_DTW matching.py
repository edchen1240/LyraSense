"""
[LRS-06_convert to chroma.py]
Purpose: Register new songs into the LyraSense system.
1. Check audio.
2. Convert lyrics (LRC/STR) to carnonical json format.
3. (Other checks TBD)
Author: Meng-Chi Ed Chen
Date: 
Reference:
    1.
    2.

Status: Working.
"""
import os, sys, cv2, time
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime
import matplotlib.pyplot as plt

import LRS_M1_File as M1F
import LRS_M2_Data as M2D
import LRS_M3_DTW as M3D

#[1] Settings and parameters
dir_REF     = r'D:\05_Datasets\05_Song Data\02_LyraSense Ref'
dir_output  = r'D:\01_Floor\a_Ed\09_EECS\10_Python\03_Developing\2026-0214_LyraSense\LRS-05_Output'
track_ID_1    = '96OADL6F-96OA74'
track_ID_2    = '96OADL6F-21BAF3'
track_ID_3    = '96OADL6F-19MPC9'


#[2] Initialize DTWMatching object
DTW_run_settings = {
    'add_DTW_path':         True,
    'key_correction':       True,   # Whether to apply key correction before DTW (e.g., via chroma cross-correlation)
}

#[4] Initialize DTWMatching object with the two tracks to compare
DTWM = M3D.DTWMatching( dir_REF, 
                        dir_output, 
                        track_ID_1, 
                        track_ID_3,
                        DTW_run_settings)

df_options = pd.DataFrame({
    'Option': ['A. Perform DTW matching', 
               'B. Change track_ID', 
               'C. -- TBD --', 
               'D. -- TBD --'],
    'Description': [
    'Perform DTW matching between the two tracks and visualize the similarity matrix.',
    'Change the track_IDs to compare different pairs of tracks.',
    '-- TBD --',
    '-- TBD --'
    ]})

text_q1 =   f'\nWelcome to the LyraSense DTW Matching Tool.\n'\
            f'This tool allows you to perform Dynamic Time Warping (DTW) matching between two tracks and visualize the similarity matrix.\n'\
            f'\nPlease choose an option (track_ID = work_ID + recording_ID):\n'\
            f'{tabulate(df_options, headers="keys", tablefmt="simple", showindex=False)}\n'
q1_ans = M1F.ask_for_input(text_q1, True, False)



if q1_ans.upper() == 'A':

    # Setting for DTW are writtne in LRS_M3_DTW.py > DTW_SETTINGS.
    DTWM.compute_similarity_matrix()
    DTWM.compute_optimal_path()
    DTWM.visualize_similarity_matrix()
    
elif q1_ans.upper() == 'B':
    text_q2_2 = f'Change track_ID_1 from {track_ID_1} to what? (or press enter to skip)\n'
    _track_ID_1 = M1F.ask_for_input(text_q2_2, False, True)
    if _track_ID_1 != '':
        track_ID_1 = _track_ID_1
    text_q2_3 = f'Change track_ID_2 from {track_ID_2} to what? (or press enter to skip)\n'
    _track_ID_2 = M1F.ask_for_input(text_q2_3, False, True)
    if _track_ID_2 != '':
        track_ID_2 = _track_ID_2
    DTWM = M3D.DTWMatching(dir_REF, 
                       dir_output, 
                       track_ID_1, 
                       track_ID_2)
else:
    raise NotImplementedError(f'Option {q1_ans} is not implemented yet.')



    
#[20] Complete
print(f'\nCompleted {os.path.basename(__file__)}. Close in 5 seconds.')
time.sleep(5)
sys.exit()




















