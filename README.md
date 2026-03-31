# LyraSense
Real-Time Audio–Lyric Alignment for Live Music Performance

LyraSense is a real-time lyrics-following system that continuously aligns a
live audio stream to lyric text.  Given a live audio stream containing
singing or instrumental accompaniment, it identifies the underlying song from a
confined song bank and tracks the current lyric line with low latency.

## How It Works

1. **Song Identification** – A short audio probe (~3 s) is compared against
   reference features for every song in the bank using sliding-window cosine
   similarity on chroma features.
2. **Real-time Alignment** – An Online DTW (Dynamic Time Warping) aligner with
   a Sakoe–Chiba band constraint maps each incoming audio frame to a position
   in the reference song, then converts that position to the corresponding
   lyric line.
3. **Streaming Design** – Audio is processed in small chunks (~93 ms at 22 050 Hz)
   so end-to-end latency stays well below one second on commodity hardware.
4. **Chorus Handling** – Because DTW finds the *best* alignment position across
   the entire reference, repeated structures such as choruses are handled
   naturally without special-casing.

## Project Structure

```
lyrasense/
├── audio/
│   ├── capture.py       # Real-time microphone capture (sounddevice)
│   └── features.py      # Chroma / MFCC extraction (librosa), L2-normalised
├── song_bank/
│   ├── song.py          # LyricLine & Song data structures
│   └── manager.py       # SongBank: load songs from JSON / raw audio
├── identification/
│   └── identifier.py    # Cosine-similarity-based song identifier
├── alignment/
│   ├── dtw.py           # Online DTW (streaming, Sakoe–Chiba band)
│   └── tracker.py       # LyricTracker: chunk → LyricLine pipeline
├── core.py              # LyraSense orchestrator
└── cli.py               # Command-line interface
data/example_songs/      # Example song JSON descriptors
tests/                   # 44 pytest tests
```

## Installation

```bash
pip install -e ".[dev]"
```

Dependencies: `numpy`, `scipy`, `librosa`, `sounddevice`.

## Usage

### Live tracking (microphone)

```bash
lyrasense track --bank data/example_songs/
```

### Offline simulation (WAV file)

```bash
lyrasense align --song data/example_songs/demo.json --audio recording.wav
```

### List songs in the bank

```bash
lyrasense list --bank data/example_songs/
```

### Python API

```python
from lyrasense import LyraSense, SongBank

bank = SongBank()
bank.add_song_from_file("data/example_songs/demo.json")

def on_lyric(line):
    print(f"♪  {line.text}")

ls = LyraSense(bank, lyric_callback=on_lyric)
ls.start()   # begins microphone capture
# ... press Ctrl-C to stop
ls.stop()
```

## Song Bank Format

Songs are described by JSON files:

```json
{
    "title": "My Song",
    "artist": "Artist Name",
    "sample_rate": 22050,
    "feature_hop_length": 512,
    "lyrics": [
        {"text": "First lyric line",  "start_time": 0.0,  "end_time": 3.0},
        {"text": "Second lyric line", "start_time": 3.0,  "end_time": 6.5}
    ],
    "reference_features": "features.npy"
}
```

`reference_features` can be a path to a `.npy` file (pre-computed chroma
matrix), an inline 2-D list, or `null` (features are computed on first use
from audio added programmatically via `SongBank.add_song_from_audio()`).

## Running Tests

```bash
python -m pytest tests/ -v
```

