"""
LyraSense – Real-Time Audio–Lyric Alignment for Live Music Performance.

Public API::

    from lyrasense import LyraSense, SongBank

    bank = SongBank()
    bank.add_song_from_file("song.json")

    ls = LyraSense(bank)
    ls.start()          # begins microphone capture
    ls.stop()
"""

from lyrasense.core import LyraSense
from lyrasense.song_bank.manager import SongBank

__all__ = ["LyraSense", "SongBank"]
__version__ = "0.1.0"
