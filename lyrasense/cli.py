"""Command-line interface for LyraSense.

Usage examples::

    # Start live lyric tracking using a song bank directory
    lyrasense track --bank data/example_songs/

    # Run an offline alignment simulation on a WAV file
    lyrasense align --song data/example_songs/demo.json --audio recording.wav

    # List all songs in the bank
    lyrasense list --bank data/example_songs/
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from lyrasense.core import LyraSense
from lyrasense.song_bank.manager import SongBank
from lyrasense.song_bank.song import LyricLine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_bank(bank_dir: str) -> SongBank:
    """Load all *.json song descriptors from *bank_dir*."""
    bank = SongBank()
    bank_path = Path(bank_dir)
    json_files = sorted(bank_path.glob("*.json"))
    if not json_files:
        print(f"[warn] No .json files found in {bank_dir}", file=sys.stderr)
    for jf in json_files:
        try:
            song = bank.add_song_from_file(jf)
            print(f"  Loaded: {song.title!r} by {song.artist!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] Failed to load {jf}: {exc}", file=sys.stderr)
    return bank


def _lyric_display_callback(line: LyricLine) -> None:
    """Print a newly active lyric line to stdout."""
    print(f"\033[32m♪  {line.text}\033[0m")


# ---------------------------------------------------------------------------
# Subcommand: track
# ---------------------------------------------------------------------------


def cmd_track(args: argparse.Namespace) -> int:
    """Start live real-time lyric tracking from the microphone."""
    print("Loading song bank…")
    bank = _load_bank(args.bank)
    if len(bank) == 0:
        print("Error: No songs loaded.  Aborting.", file=sys.stderr)
        return 1

    print(f"\nLoaded {len(bank)} song(s).  Starting microphone capture…")
    print("Press Ctrl+C to stop.\n")

    ls = LyraSense(
        song_bank=bank,
        lyric_callback=_lyric_display_callback,
        id_probe_seconds=args.probe,
    )

    def _handle_signal(sig: int, frame: object) -> None:
        print("\nStopping…")
        ls.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    ls.start()
    print(f"Listening… (identification probe: {args.probe:.1f}s)")
    while ls.is_running:
        time.sleep(0.1)

    return 0


# ---------------------------------------------------------------------------
# Subcommand: align
# ---------------------------------------------------------------------------


def cmd_align(args: argparse.Namespace) -> int:
    """Offline simulation: feed a WAV file through LyraSense and print lyrics."""
    try:
        import librosa  # type: ignore
    except ImportError:
        print("Error: librosa is required for offline alignment.  pip install librosa", file=sys.stderr)
        return 1

    print(f"Loading song descriptor: {args.song}")
    bank = SongBank()
    try:
        song = bank.add_song_from_file(args.song)
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading song: {exc}", file=sys.stderr)
        return 1

    print(f"Loading audio: {args.audio}")
    try:
        audio, sr = librosa.load(args.audio, sr=song.sample_rate, mono=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading audio: {exc}", file=sys.stderr)
        return 1

    if song.reference_features is None:
        print("Computing reference features from audio descriptor… (no pre-built .npy)")

    ls = LyraSense(song_bank=bank, lyric_callback=_lyric_display_callback)
    ls.force_song(song)

    chunk_size = 2048
    print(f"\n--- Simulated real-time playback of {args.audio} ---\n")
    last_line = None
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        line = ls.process_chunk(chunk)
        if line is not last_line and line is not None:
            elapsed = start / sr
            print(f"[{elapsed:6.2f}s] {line.text}")
            last_line = line

    print("\n--- Alignment complete ---")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    """List all songs in a bank directory."""
    bank = _load_bank(args.bank)
    if not bank.all_songs():
        print("No songs found.")
        return 0

    print(f"\n{'#':<4} {'Title':<30} {'Artist':<20} {'Lines':<6} {'Features'}")
    print("-" * 72)
    for i, song in enumerate(bank.all_songs(), 1):
        has_feat = "yes" if song.reference_features is not None else "no"
        print(f"{i:<4} {song.title:<30} {song.artist:<20} {len(song.lyrics):<6} {has_feat}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lyrasense",
        description="LyraSense – Real-Time Audio–Lyric Alignment for Live Music Performance",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # track
    p_track = sub.add_parser("track", help="Live real-time lyric tracking.")
    p_track.add_argument(
        "--bank",
        default="data/example_songs",
        metavar="DIR",
        help="Directory containing song JSON descriptors (default: data/example_songs).",
    )
    p_track.add_argument(
        "--probe",
        type=float,
        default=3.0,
        metavar="SECS",
        help="Seconds of audio to collect for song identification (default: 3.0).",
    )

    # align
    p_align = sub.add_parser("align", help="Offline alignment simulation on a WAV file.")
    p_align.add_argument("--song", required=True, metavar="JSON", help="Song descriptor JSON.")
    p_align.add_argument("--audio", required=True, metavar="WAV", help="Audio file to align.")

    # list
    p_list = sub.add_parser("list", help="List songs in the bank.")
    p_list.add_argument(
        "--bank",
        default="data/example_songs",
        metavar="DIR",
        help="Bank directory (default: data/example_songs).",
    )

    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=log_level,
    )

    if args.command == "track":
        sys.exit(cmd_track(args))
    elif args.command == "align":
        sys.exit(cmd_align(args))
    elif args.command == "list":
        sys.exit(cmd_list(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
