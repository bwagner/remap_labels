#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pytest",
# ]
# ///
"""Tests for gen_grids command construction."""

import pytest


def test_build_command_default_no_beats_per_bar():
    from gen_grids import _build_dbn_command

    cmd = _build_dbn_command("song.mp3")
    assert cmd == ["DBNDownBeatTracker", "single", "song.mp3"]


def test_build_command_single_beats_per_bar():
    from gen_grids import _build_dbn_command

    cmd = _build_dbn_command("song.mp3", beats_per_bar="5")
    assert cmd == ["DBNDownBeatTracker", "--beats_per_bar", "5", "single", "song.mp3"]


def test_build_command_list_beats_per_bar():
    from gen_grids import _build_dbn_command

    cmd = _build_dbn_command("song.mp3", beats_per_bar="3,4,5")
    assert cmd == ["DBNDownBeatTracker", "--beats_per_bar", "3,4,5", "single", "song.mp3"]


def test_build_command_empty_string_treated_as_none():
    from gen_grids import _build_dbn_command

    cmd = _build_dbn_command("song.mp3", beats_per_bar="")
    assert cmd == ["DBNDownBeatTracker", "single", "song.mp3"]


def test_build_command_flag_placed_before_subcommand():
    """DBN requires --beats_per_bar BEFORE the 'single' subcommand."""
    from gen_grids import _build_dbn_command

    cmd = _build_dbn_command("song.mp3", beats_per_bar="4")
    assert cmd.index("--beats_per_bar") < cmd.index("single"), (
        f"--beats_per_bar must precede 'single', got {cmd}"
    )
