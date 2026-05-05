# LyraSense — Implementation Spec (May 2026 Restart)

*Design decisions finalized in conversation. Reference this when coding.*

---

## 1. Project Status Summary

| Component | Module(s) | Status |
|-----------|-----------|--------|
| Song registration pipeline | `LRS-05`, `LRS-06` | **Done** |
| Chroma extraction (CQT, offline) | `LRS_M2_Data.py` | **Done** |
| Batch DTW (full-matrix, Sakoe–Chiba) | `LRS_M3_DTW.py`, `LRS-12` | **Done** |
| Paper draft (Sections 1–6) | `03_final_project.tex` | **Done** |
| LRC parser → canonical JSON | — | **Not started** |
| Lyrics timestamp shift tool | — | **Not started** |
| Audio stream loader | — | **Not started** |
| Online/incremental DTW | — | **Not started** |
| Lyric lookup (t_ref → line) | — | **Not started** |
| Evaluation harness | — | **Not started** |
| Microphone integration | — | **Not started** |

---

## 2. LRC Parser (`LRS-07` or integrate into `LRS-05`)

### Scope
- **Input format**: Simple LRC only — `[mm:ss.xx]text` per line.
- **No** enhanced LRC (word-level `<mm:ss.xx>` tags), SRT, or VTT for now.

### Output schema (canonical JSON)
```json
{
  "work_id": "96OADL6F",
  "recording_id": "96OA74",
  "lines": [
    {
      "line_id": 0,
      "start_time": 12.34,
      "end_time": 17.89,
      "text": "Slip inside the eye of your mind",
      "part": "verse_1"
    }
  ]
}
```

### Key rules
- `end_time` = next line's `start_time` (inferred).
- Last line's `end_time` = audio file duration (from reference metadata).
- Strip LRC metadata tags (`[ar:`, `[al:`, `[ti:`, `[offset:`, etc.).
- Skip blank/instrumental lines (`[mm:ss.xx]` with no text or only whitespace).
- `part` field populated manually or left as `null` initially.
- Timestamp regex: `\[(\d{2}):(\d{2})\.(\d{2,3})\](.*)`.

---

## 3. Lyrics Timestamp Shift Tool

### Two-pass workflow

**Pass 1 — Global offset (script)**
- Input: canonical JSON + offset in seconds (float, positive = shift later).
- Operation: `new_start = old_start + offset` for all lines. Clamp to ≥ 0.
- Output: overwrite or save as new JSON.
- CLI: `python lrs_lyrics_shift.py --json path/to/lyrics.json --offset -1.5`

**Pass 2 — Per-line fine-tuning (terminal-based)**
- For each line:
  1. Play audio from `start_time - 2s` to `start_time + 3s` (using `sounddevice` or `simpleaudio`).
  2. Print: `Line {line_id}: "{text}" @ {start_time:.2f}s`
  3. Prompt: `Offset (sec, or Enter to keep, 'r' to replay, 'q' to quit):`
  4. Apply per-line correction.
- Save after each line (crash-safe).
- Allow jumping to a specific `line_id` to resume.

### Environment
- Terminal-based. No GUI dependencies (no Tkinter, no Flask).
- Audio playback via `sounddevice.play()` (already needed for mic integration).

---

## 4. Audio Stream Loader (`LRS_M4_Stream.py`)

### Architecture: dual-backend, common ABC

```
                    ┌─────────────────────┐
                    │  AudioSource (ABC)  │
                    │  ─────────────────  │
                    │  get_chunk() → np   │
                    │  is_active() → bool │
                    │  close()            │
                    └──────┬──────┬───────┘
                           │      │
            ┌──────────────┘      └──────────────┐
            ▼                                    ▼
 ┌─────────────────────┐            ┌─────────────────────┐
 │  FileAudioSource    │            │  MicAudioSource     │
 │  (Branch A)         │            │  (Branch B)         │
 │  ─────────────────  │            │  ─────────────────  │
 │  Poll-pull          │            │  Callback-push      │
 │  Deterministic      │            │  Thread-safe queue  │
 │  get_chunk() blocks │            │  get_chunk() = pop  │
 │  for chunk_duration │            │  from queue         │
 └─────────────────────┘            └─────────────────────┘
```

### Design decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | **1 second (44100 samples)** | ~43 chroma frames/chunk. Clean CQT with `fmin=32.7 Hz`. Balanced latency vs. stability. |
| Sample rate | 44100 Hz | Matches `CHROMA_SETTINGS` in `M2D` |
| Channels | Mono | Downmix stereo on ingest |
| Dtype | `float32` | librosa convention |

### `FileAudioSource` (Branch A — pseudo-real-time)
- **Interface**: Poll-pull. `get_chunk()` reads next 44100 samples from the file, then `time.sleep(chunk_duration)` to simulate real-time pacing.
- **Use case**: Pseudo-RT evaluation (deterministic, reproducible, debuggable).
- **End-of-file**: `get_chunk()` returns `None` or raises `StopIteration`.
- **No threading** — single-threaded, sequential.

### `MicAudioSource` (Branch B — live microphone)
- **Interface**: Callback-push. `sounddevice.InputStream` fires callback → appends chunk to `queue.Queue(maxsize=N)`.
- **`get_chunk()`**: Pops from queue with timeout. Returns `None` on timeout (silence/underrun).
- **Use case**: Live demo, real-time alignment.
- **Thread safety**: `queue.Queue` handles producer (audio thread) / consumer (main thread) synchronization.
- **Overflow policy**: If queue is full, log warning and drop oldest chunk (ring-buffer semantics).

### Chunk overlap
- To avoid CQT edge artifacts at chunk boundaries, maintain a rolling buffer that retains the last `overlap_samples` from the previous chunk.
- Suggested overlap: 50% (22050 samples). This means each CQT computation sees 2 seconds of audio context but advances by 1 second.
- The chroma output is trimmed to only the new frames (frames corresponding to the non-overlapping portion).

---

## 5. Lyric Lookup Module

### Function signature
```python
def lookup_lyric(t_ref: float, lyrics: list[dict]) -> dict | None:
    """
    Given a reference time position t_ref (seconds) and the canonical
    lyrics list, return the active lyric line dict, or None if t_ref
    is outside all line intervals.
    
    Uses bisect on start_time for O(log n) lookup.
    """
```

### Implementation notes
- Pre-sort `lyrics` by `start_time` (should already be sorted from LRC).
- Use `bisect.bisect_right(start_times, t_ref) - 1` to find candidate line.
- Verify `t_ref < candidate.end_time` (otherwise we're in a gap or past the end).
- Return current line + optionally next 1–2 lines for display lookahead.

---

## 6. Build Order (Critical Path)

```
Phase 1 (unblock Stage IV — can start immediately)
  ├── LRC parser → canonical JSON
  ├── Lyric lookup module (t_ref → line)
  └── Lyrics timestamp shift tool (global + per-line)

Phase 2 (go online — depends on Phase 1 + existing DTW)
  ├── LRS_M4_Stream.py (FileAudioSource + MicAudioSource)
  ├── Online DTW (incremental cost matrix, sliding column)
  └── Evaluation harness (pseudo-RT metrics)

Phase 3 (live demo — depends on Phase 2)
  ├── End-to-end mic → chroma → DTW → lyrics pipeline
  └── Live display (scrolling lyrics + confidence)
```

---

## 7. Open Questions for Later

- **Online DTW algorithm choice**: Dixon (2005) "An On-Line Time Warping Algorithm" vs. subsequence DTW from Müller Ch. 4. Dixon is simpler but assumes monotonic tempo; subsequence DTW handles mid-song entry.
- **Song identification**: Currently assumes known song. Multi-reference matching (compare incoming chroma against all references simultaneously) deferred to Phase 3.
- **Evaluation dataset size**: How many songs × how many versions in the test set? Currently only *Don't Look Back in Anger* × 3 versions validated.
