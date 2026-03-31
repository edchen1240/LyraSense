"""Tests for the CLI entry point (offline simulation)."""
from __future__ import annotations

import json
import tempfile
import pathlib

import numpy as np
import pytest

from lyrasense.cli import main


class TestCLIList:
    def test_list_empty_dir(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["list", "--bank", str(tmp_path)])
        assert exc_info.value.code == 0

    def test_list_with_song(self, tmp_path: pathlib.Path) -> None:
        data = {
            "title": "CLI Song",
            "artist": "CLI Artist",
            "lyrics": [{"text": "Hello", "start_time": 0.0, "end_time": 2.0}],
        }
        (tmp_path / "cli_song.json").write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            main(["list", "--bank", str(tmp_path)])
        assert exc_info.value.code == 0


class TestCLIHelp:
    def test_no_command_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0
