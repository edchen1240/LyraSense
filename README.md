
# LyraSense
Real-Time Audio–Lyric Alignment for Live Music Performance



- Last Updated: 2026-02-15, 09:27
- Last Updated: 2026-03-30, 22:56

This file records the technical design and workflow for the LyraSense reference data schema.




# Work Flow


Audio 
→ Feature Timeline (with Chroma)
→ Temporal Alignment (with DTW)
→ Lyric Mapping



# V. Data Structure

This document defines the reference data schema for LyraSense.  
It is intended for both human contributors and LLM-assisted development.


## 1. Design Principles

- Data is hierarchical: **song > version > line**
- Metadata is stored at the smallest level it describes.
- Numeric arrays are stored in binary format (NPZ).
- Structured metadata is stored in JSON.
- Files are machine-optimized; human inspection is done via scripts.


## 2. Directory Architecture (PoC Stage)
```
reference/
├── work_ID-001/
│   ├── work_ID_work meta.json
│   ├── version-ID_01/
│   │   ├── recording_ID-01.mp3
│   │   ├── recording_ID-01.npz
│   │   └── recording_ID-01_recording meta.json
│   └── version-ID_02/
│       ├── recording_ID-02.mp3
│       ├── recording_ID-02.npz
│       └── recording_ID-02_recording meta.json
├── work_ID-002/
│   └── ...
```

## 3. Naming Conventions

### Plan | track_ID = work_ID + recording_ID


- Every abstract composition (melody + lyrics. Same song but different recording version.)
    is assigned with a 6-character work_ID.
- Every version (covers, live versions, remasters) of that song has a 6-character recording_ID. 
- Together they are called track_ID, a 12-character ID.
- We avoid "song" since it might be ambiguous in our context.
- Same song but different version will have the same track_ID but different recording_ID.




## 3. Reference Audio Features

Reference audio is pre-rendered from MP3/WAV into chroma features.

- Format: `.npz`
- Shape: `(12, T)`
- Content:
  - `chroma`: chroma matrix
  - `times`: frame timestamps (seconds)

Chroma features are used for DTW-based alignment.  
NPZ is chosen for fast loading, compact storage, and numerical precision.

---

## 4. Lyrics Data (Version-Level)

Each `version_XX.json` contains:

```json
{
  "version_meta": {
    "sr": 44100,
    "hop_length": 512,
    "frame_duration": 0.0116,
    "starting key": "C",     # Song might have key modulation. Not critical for PoC but may be useful for future phoneme-level alignment.
    "tempo_estimate": 74
  },
  "lyrics": [
    {
      "line_id": 0,
      "start_time": 5.32,
      "end_time": 9.84,
      "text": "Imagine there's no heaven",
      "part": "verse"
    }
  ]
}
```

## 5. Current Line-Level Fields
- line_id: unique identifier within version
- start_time: start time in seconds
- end_time: end time in seconds
- text: lyric string
- part: structural label (intro, verse, chorus, etc.)
JSON is used instead of CSV/Excel because:
- It supports hierarchical metadata (song + version + line).
- It avoids delimiter/encoding issues in lyric text.
- It maps cleanly to future HDF5 migration.
- It allows schema evolution (phonemes, translations, confidence, etc.).
Performance differences between CSV and JSON are negligible at this scale.
Excel is avoided due to parsing overhead and poor version control behavior.

## 6. Song-Level Metadata
Each song_meta.json contains:
```json
{
  "song_id": "song_001",
  "title": "Imagine",
  "composer": "John Lennon",
  "language": "en",
  "genre": "pop"
}
```
Song-level metadata is separated from version-level metadata to maintain
clear ontology and prevent cross-level coupling.

## 7. Future Scalability
When scaling beyond PoC (~100 songs):
- All data may be migrated into a single reference_bank.h5.
- Song groups → HDF5 groups.
- Version data → datasets.
- Metadata → group attributes.
- Lyrics → JSON string or structured dataset.
At large scale, storage is machine-optimized.
Human readability is provided via inspection scripts, not raw file editing.



# VI. New Song Registration Workflow

## Overview
Download audio. Align lyrics first, then generate chroma features. Lyrics might be tricky to align, so we start with them. Chroma feature would just a press of a button.
1. Download audio from YouTube. (Make sure it's a clean version with minimal background noise.)
2. Download lyrics with timestamps (LRC/SRT/VTT). 
3. Roughtly check if the lyrics timestamps is aligned with audio. (2026-0215-0941: We will build UI to make adjustment later.)
4. Convert Lyrics to Canonical JSON
5. Normalize Audio
6. Extract Reference Features




# X. Current Progress

## 2026-02-16 — Summary
1. Data model
   - `track_ID` = `work_ID` + `recording_ID` (hierarchical: composition → version).

2. NewSongRegistration (implemented)
   - Verifies directories & JSON meta files (`*work meta.json`, `*recording meta.json`).
   - Detects a single audio file per recording and extracts diagnostics (`sr`, `duration`, `channels`, `value_min/max`, `nan/inf`, `file_size`).
   - Persists recording metadata automatically to `recording meta.json`.
   - Computes & validates chroma (`<work_ID>_chroma.npz`) from audio (mono conversion, optional resample, per-frame normalization).

3. Extracted two chroma of the same song but different versions (original vs cover).

4. Status: Working — automated metadata saving and chroma extraction are available.





## 2026-02-27 — Summary
1. Implemented `create_work_meta` function to generate work-level metadata by looking up the track_ID in the Excel sheet and filling in missing keys (e.g., path to original audio).
2. Updated the workflow to check if metadata needs to be updated (e.g., missing keys, empty file, or file not exist). If so, it will call `create_work_meta` to update and save the metadata.
3. Tested the workflow with a specific track_ID, and it successfully created/updated the work_meta with the expected keys and values, including the path to the original audio if available.


TODO:
1. All work-level registration. or check.
2. DTW !!!
3. self.list_track_ID_need_update



## 2026-02-28 — Summary
1. Implemented `create_recording_meta` function to generate recording-level metadata by analyzing the audio file (e.g., sample rate, duration, channels, value range, NaN/inf presence, file size).
2. Look into self.list_track_ID_need_update





## 2026-0228-1159:
1. Improved create_work_meta.
2. create_recd_meta_and_chroma. Need to finish sys.exit('Checking the generated recording meta with the template ...').

## 2026-0602-0114:
1. Update track_ID meta validation.
2. DTW matrix and DTW path visualization.








