"""
[LRS-05_new song registration.py]
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
import os, sys, time
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime
import matplotlib.pyplot as plt


import LRS_M2_Data as M2D






dir_REF     = r'D:\05_Datasets\05_Song Data\02_LyraSense Ref'
track_ID_1    = '96OADL6F-96OA74'
track_ID_2    = '96OADL6F-21BAF3'
track_ID_3    = '96OADL6F-21MP97'

track_ID_6    = '71JDTM25-71JD09'

track_ID = track_ID_6
# track_ID = work_ID + recording_ID



NSR = M2D.NewSongRegistration(dir_REF, track_ID)





#[20] Complete
print(f'\nCompleted {os.path.basename(__file__)}. Close in 5 seconds. {track_ID}')
time.sleep(5)
sys.exit()



