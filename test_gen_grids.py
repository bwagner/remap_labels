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


def test_format_bars_event_mode_default():
    from gen_grids import _format_bars

    lines = _format_bars([1.0, 3.0, 5.0])
    assert lines == [
        "1.000000\t1.000000\t1\n",
        "3.000000\t3.000000\t2\n",
        "5.000000\t5.000000\t3\n",
    ]


def test_format_bars_span_mode():
    from gen_grids import _format_bars

    lines = _format_bars([1.0, 3.0, 5.0], span=True)
    # N downbeats -> N-1 span lines (can't know end of last bar)
    assert lines == [
        "1.000000\t3.000000\t1\n",
        "3.000000\t5.000000\t2\n",
    ]


def test_format_bars_single_downbeat_event():
    from gen_grids import _format_bars

    assert _format_bars([2.5]) == ["2.500000\t2.500000\t1\n"]


def test_format_bars_single_downbeat_span_empty():
    """A single downbeat in span mode emits nothing (no next bar to end at)."""
    from gen_grids import _format_bars

    assert _format_bars([2.5], span=True) == []


def test_format_bars_empty_downbeats():
    from gen_grids import _format_bars

    assert _format_bars([]) == []
    assert _format_bars([], span=True) == []
