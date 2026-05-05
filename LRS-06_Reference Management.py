"""
[LRS-06_Reference Management.py]
Purpose: Reference Management.
1. Manage reference data for songs, including metadata and file paths.
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






df_options = pd.DataFrame({
    'Option': ['A. Full work_ID check', 
               'B. Full track_ID check', 
               'C. Check audio files', 
               'D. New track registration'],
    'Description': [
    'Check if every unique work_ID \nhas a corresponding reference folder with complete work_meta.',
    'Check if every unique track_ID \nhas a corresponding reference folder with complete recd_meta.',
    'Check if all audio files are in mp3, mono, 44.1kHz.',
    'Register a new track \nby creating a reference folder and saving the provided metadata.'
    ]})

text_q1 =   f'\nWelcome to the LyraSense Reference Management.\n'\
            f'This tool helps you manage reference data for songs, especially work meta and recording meta.\n'\
            f'\nPlease choose an option (track_ID = work_ID + recording_ID):\n'\
            f'{tabulate(df_options, headers="keys", tablefmt="simple", showindex=False)}\n'
q1_ans = M1F.ask_for_input(text_q1, True, False)



if q1_ans.upper() == 'A':
    FWC = M2D.Full_work_ID_Check()
elif q1_ans.upper() == 'B':
    FTC = M2D.Full_track_ID_Check()
else:
    raise NotImplementedError(f'Option {q1_ans} is not implemented yet.')





#[20] Complete
print(f'\nCompleted {os.path.basename(__file__)}. Close in 5 seconds.' )
time.sleep(5)
sys.exit()



