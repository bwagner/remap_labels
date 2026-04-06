"""Tests for label_info: bar/beat display and pattern detection."""


import pytest

from remap_labels import LabelEntry


# -- Fixtures --

# Beat grid: 4 beats/bar, 0.5s per beat, starting at 1.0s
BEAT_GRID = [1.0 + i * 0.5 for i in range(32)]
# 8 bars
BAR_GRID = [BEAT_GRID[i] for i in range(0, 32, 4)]

# Larger grids for multi-section tests
BEAT_GRID_LARGE = [1.0 + i * 0.5 for i in range(128)]
BAR_GRID_LARGE = [BEAT_GRID_LARGE[i] for i in range(0, 128, 4)]


# -- discover_files --


def test_discover_from_mp3(tmp_path):
    """Given an mp3 path, discovers label, bars, and beats files."""
    from label_info import discover_files

    song = "blues_brothers_everybody"
    (tmp_path / f"{song}.mp3").touch()
    (tmp_path / f"bars_{song}.txt").touch()
    (tmp_path / f"beats_{song}.txt").touch()
    (tmp_path / f"chords_{song}.txt").touch()
    (tmp_path / f"parts_{song}.txt").touch()
    # Non-label file should be ignored
    (tmp_path / f"{song}.aup3").touch()

    result = discover_files(str(tmp_path / f"{song}.mp3"))

    assert result["bars"] == str(tmp_path / f"bars_{song}.txt")
    assert result["beats"] == str(tmp_path / f"beats_{song}.txt")
    assert sorted(result["labels"]) == sorted([
        str(tmp_path / f"chords_{song}.txt"),
        str(tmp_path / f"parts_{song}.txt"),
    ])


def test_discover_from_aup3(tmp_path):
    """Given an aup3 path, discovers the same way."""
    from label_info import discover_files

    song = "my_song"
    (tmp_path / f"{song}.aup3").touch()
    (tmp_path / f"bars_{song}.txt").touch()
    (tmp_path / f"beats_{song}.txt").touch()
    (tmp_path / f"guit_{song}.txt").touch()

    result = discover_files(str(tmp_path / f"{song}.aup3"))

    assert result["bars"] == str(tmp_path / f"bars_{song}.txt")
    assert result["beats"] == str(tmp_path / f"beats_{song}.txt")
    assert result["labels"] == [str(tmp_path / f"guit_{song}.txt")]


def test_discover_missing_bars_raises(tmp_path):
    """Raises if bars file not found."""
    from label_info import discover_files

    song = "my_song"
    (tmp_path / f"{song}.mp3").touch()
    (tmp_path / f"beats_{song}.txt").touch()
    (tmp_path / f"chords_{song}.txt").touch()

    with pytest.raises(FileNotFoundError, match="bars"):
        discover_files(str(tmp_path / f"{song}.mp3"))


def test_discover_missing_beats_raises(tmp_path):
    """Raises if beats file not found."""
    from label_info import discover_files

    song = "my_song"
    (tmp_path / f"{song}.mp3").touch()
    (tmp_path / f"bars_{song}.txt").touch()
    (tmp_path / f"chords_{song}.txt").touch()

    with pytest.raises(FileNotFoundError, match="beats"):
        discover_files(str(tmp_path / f"{song}.mp3"))


def test_discover_no_labels_raises(tmp_path):
    """Raises if no label files found."""
    from label_info import discover_files

    song = "my_song"
    (tmp_path / f"{song}.mp3").touch()
    (tmp_path / f"bars_{song}.txt").touch()
    (tmp_path / f"beats_{song}.txt").touch()

    with pytest.raises(FileNotFoundError, match="label"):
        discover_files(str(tmp_path / f"{song}.mp3"))


# -- prefix_from_filename --


def test_prefix_from_filename():
    from label_info import prefix_from_filename

    assert prefix_from_filename("chords_my_song.txt", "my_song") == "chords"
    assert prefix_from_filename("parts_my_song.txt", "my_song") == "parts"
    assert prefix_from_filename("guit_my_song.txt", "my_song") == "guit"


# -- label_to_bar_beat --


def test_label_on_bar_boundary():
    """Label starting on a bar boundary shows just the bar number."""
    from label_info import label_to_bar_beat

    # Bar 1 starts at 1.0
    result = label_to_bar_beat(1.0, BAR_GRID, BEAT_GRID)
    assert result == "bar 1"


def test_label_on_beat_2():
    """Label on beat 2 of bar 1 shows bar and beat."""
    from label_info import label_to_bar_beat

    # Beat 2 of bar 1 is at 1.5
    result = label_to_bar_beat(1.5, BAR_GRID, BEAT_GRID)
    assert result == "bar 1.2"


def test_label_on_beat_3():
    """Label on beat 3 of bar 2."""
    from label_info import label_to_bar_beat

    # Bar 2 starts at 3.0, beat 3 is at 4.0
    result = label_to_bar_beat(4.0, BAR_GRID, BEAT_GRID)
    assert result == "bar 2.3"


def test_label_show_beats_always():
    """With show_beats=True, always show beat even on bar boundary."""
    from label_info import label_to_bar_beat

    result = label_to_bar_beat(1.0, BAR_GRID, BEAT_GRID, show_beats=True)
    assert result == "bar 1.1"


# -- format_track --


def test_format_track_simple():
    """Format a simple chord track with bar positions."""
    from label_info import format_track

    labels = [
        LabelEntry(1.0, 3.0, "C"),   # bar 1, 1 bar long
        LabelEntry(3.0, 5.0, "F"),   # bar 2, 1 bar long
        LabelEntry(5.0, 7.0, "Am"),  # bar 3, 1 bar long
        LabelEntry(7.0, 9.0, "G"),   # bar 4, 1 bar long
    ]
    lines = format_track(labels, BAR_GRID, BEAT_GRID)
    assert lines == [
        "bar 1: C",
        "bar 2: F",
        "bar 3: Am",
        "bar 4: G",
    ]


def test_format_track_sub_bar():
    """Labels at sub-bar positions show beat."""
    from label_info import format_track

    labels = [
        LabelEntry(1.0, 1.5, "C"),   # bar 1 beat 1
        LabelEntry(1.5, 2.0, "D"),   # bar 1 beat 2
    ]
    lines = format_track(labels, BAR_GRID, BEAT_GRID)
    assert lines == [
        "bar 1: C",
        "bar 1.2: D",
    ]


def test_format_track_point_label():
    """Point labels (start == end) are formatted the same way."""
    from label_info import format_track

    labels = [
        LabelEntry(3.0, 3.0, "marker"),  # bar 2 point label
    ]
    lines = format_track(labels, BAR_GRID, BEAT_GRID)
    assert lines == [
        "bar 2: marker",
    ]


# -- interleave_tracks --


def test_interleave_two_tracks_no_prefix():
    """Interleave without prefix by default."""
    from label_info import interleave_tracks

    tracks = {
        "chords": [
            LabelEntry(1.0, 3.0, "C"),
            LabelEntry(3.0, 5.0, "F"),
        ],
        "parts": [
            LabelEntry(1.0, 5.0, "verse"),
        ],
    }
    lines = interleave_tracks(tracks, BAR_GRID, BEAT_GRID)
    assert lines == [
        "bar 1: C verse",
        "bar 2: F",
    ]


def test_interleave_two_tracks_with_prefix():
    """Interleave with show_prefix=True adds (prefix) suffix."""
    from label_info import interleave_tracks

    tracks = {
        "chords": [
            LabelEntry(1.0, 3.0, "C"),
            LabelEntry(3.0, 5.0, "F"),
        ],
        "parts": [
            LabelEntry(1.0, 5.0, "verse"),
        ],
    }
    lines = interleave_tracks(tracks, BAR_GRID, BEAT_GRID, show_prefix=True)
    assert lines == [
        "bar 1: C (chords) verse (parts)",
        "bar 2: F (chords)",
    ]


def test_interleave_no_overlap():
    """Non-overlapping tracks list labels in bar order."""
    from label_info import interleave_tracks

    tracks = {
        "chords": [
            LabelEntry(3.0, 5.0, "F"),
        ],
        "parts": [
            LabelEntry(1.0, 3.0, "intro"),
        ],
    }
    lines = interleave_tracks(tracks, BAR_GRID, BEAT_GRID)
    assert lines == [
        "bar 1: intro",
        "bar 2: F",
    ]


# -- detect_pattern --


def test_detect_simple_repeat():
    """Detect a 2-bar pattern repeated 3 times."""
    from label_info import detect_pattern

    # C F | C F | C F  (bars 1-2, 3-4, 5-6)
    labels = [
        LabelEntry(1.0, 3.0, "C"),   # bar 1
        LabelEntry(3.0, 5.0, "F"),   # bar 2
        LabelEntry(5.0, 7.0, "C"),   # bar 3
        LabelEntry(7.0, 9.0, "F"),   # bar 4
        LabelEntry(9.0, 11.0, "C"),  # bar 5
        LabelEntry(11.0, 13.0, "F"), # bar 6
    ]
    segments = detect_pattern(labels, BAR_GRID, BEAT_GRID)
    repeats = [s for s in segments if s["type"] == "repeat"]
    assert len(repeats) == 1
    p = repeats[0]
    assert p["count"] == 3
    assert p["bars"] == 2
    assert p["labels"] == ["C", "F"]


def test_detect_smallest_unit():
    """Prefer smallest repeating unit over larger multiples."""
    from label_info import detect_pattern

    # C F | Bb F repeated 4 times = 8 bars
    # Should find 2-bar pattern x4, not 4-bar pattern x2
    labels = [
        LabelEntry(1.0, 1.5, "C"),    # bar 1 beat 1
        LabelEntry(1.5, 2.0, "F"),    # bar 1 beat 2 (sub-bar)
        LabelEntry(3.0, 3.5, "Bb"),   # bar 2 beat 1
        LabelEntry(3.5, 4.0, "F"),    # bar 2 beat 2 (sub-bar)
        LabelEntry(5.0, 5.5, "C"),    # bar 3
        LabelEntry(5.5, 6.0, "F"),
        LabelEntry(7.0, 7.5, "Bb"),   # bar 4
        LabelEntry(7.5, 8.0, "F"),
        LabelEntry(9.0, 9.5, "C"),    # bar 5
        LabelEntry(9.5, 10.0, "F"),
        LabelEntry(11.0, 11.5, "Bb"), # bar 6
        LabelEntry(11.5, 12.0, "F"),
        LabelEntry(13.0, 13.5, "C"),  # bar 7
        LabelEntry(13.5, 14.0, "F"),
        LabelEntry(15.0, 15.5, "Bb"), # bar 8
        LabelEntry(15.5, 16.0, "F"),
    ]
    segments = detect_pattern(labels, BAR_GRID, BEAT_GRID)
    repeats = [s for s in segments if s["type"] == "repeat"]
    assert len(repeats) == 1
    p = repeats[0]
    assert p["bars"] == 2
    assert p["count"] == 4
    assert p["labels"] == ["C", "F", "Bb", "F"]


def test_detect_no_repeat():
    """No repeating pattern when all labels differ."""
    from label_info import detect_pattern

    labels = [
        LabelEntry(1.0, 3.0, "C"),
        LabelEntry(3.0, 5.0, "F"),
        LabelEntry(5.0, 7.0, "Am"),
        LabelEntry(7.0, 9.0, "G"),
    ]
    segments = detect_pattern(labels, BAR_GRID, BEAT_GRID)
    repeats = [s for s in segments if s["type"] == "repeat"]
    assert len(repeats) == 0


def test_detect_pattern_with_remainder():
    """Pattern repeated with leftover bars reported as literal."""
    from label_info import detect_pattern

    # C F | C F | Am  (2-bar pattern 2x, then 1 bar remainder)
    labels = [
        LabelEntry(1.0, 3.0, "C"),
        LabelEntry(3.0, 5.0, "F"),
        LabelEntry(5.0, 7.0, "C"),
        LabelEntry(7.0, 9.0, "F"),
        LabelEntry(9.0, 11.0, "Am"),
    ]
    segments = detect_pattern(labels, BAR_GRID, BEAT_GRID)
    repeats = [s for s in segments if s["type"] == "repeat"]
    literals = [s for s in segments if s["type"] == "literal"]
    assert len(repeats) == 1
    assert repeats[0]["count"] == 2
    assert repeats[0]["labels"] == ["C", "F"]
    assert len(literals) == 1
    assert literals[0]["bar_tuples"] == [("Am",)]


def test_detect_multiple_sections():
    """Detect repeat, then different bars, then another repeat."""
    from label_info import detect_pattern

    # C F | C F | Am | G | C F | C F
    labels = [
        LabelEntry(1.0, 3.0, "C"),
        LabelEntry(3.0, 5.0, "F"),
        LabelEntry(5.0, 7.0, "C"),
        LabelEntry(7.0, 9.0, "F"),
        LabelEntry(9.0, 11.0, "Am"),
        LabelEntry(11.0, 13.0, "G"),
        LabelEntry(13.0, 15.0, "C"),
        LabelEntry(15.0, 17.0, "F"),
    ]
    # Extended beat/bar grids for 8+ bars
    beat_grid = [1.0 + i * 0.5 for i in range(64)]
    bar_grid = [beat_grid[i] for i in range(0, 64, 4)]

    segments = detect_pattern(labels, bar_grid, beat_grid)
    types = [s["type"] for s in segments]
    assert types == ["repeat", "literal", "literal", "literal", "literal"]
    assert segments[0]["count"] == 2
    assert segments[0]["labels"] == ["C", "F"]


# -- _print_interleave_compact --


def test_interleave_compact_splits_at_section(capsys):
    """Repeat splits at non-primary labels, annotated with brackets."""
    from label_info import _print_interleave_compact

    # C | F repeated 4x = 8 bars, parts label at bar 1 and bar 5
    tracks = {
        "chords": [
            LabelEntry(1.0, 3.0, "C"),
            LabelEntry(3.0, 5.0, "F"),
            LabelEntry(5.0, 7.0, "C"),
            LabelEntry(7.0, 9.0, "F"),
            LabelEntry(9.0, 11.0, "C"),
            LabelEntry(11.0, 13.0, "F"),
            LabelEntry(13.0, 15.0, "C"),
            LabelEntry(15.0, 17.0, "F"),
        ],
        "parts": [
            LabelEntry(1.0, 9.0, "verse1"),
            LabelEntry(9.0, 17.0, "verse2"),
        ],
    }
    _print_interleave_compact(
        tracks, BAR_GRID_LARGE, BEAT_GRID_LARGE,
    )
    output = capsys.readouterr().out
    lines = [ln.strip() for ln in output.strip().splitlines()]
    assert "bars 1-4: | C | F | x2  [verse1]" in lines
    assert "bars 5-8: | C | F | x2  [verse2]" in lines


def test_interleave_compact_no_extra_labels(capsys):
    """Without non-primary labels, repeat is not split."""
    from label_info import _print_interleave_compact

    tracks = {
        "chords": [
            LabelEntry(1.0, 3.0, "C"),
            LabelEntry(3.0, 5.0, "F"),
            LabelEntry(5.0, 7.0, "C"),
            LabelEntry(7.0, 9.0, "F"),
        ],
    }
    _print_interleave_compact(
        tracks, BAR_GRID_LARGE, BEAT_GRID_LARGE,
    )
    output = capsys.readouterr().out
    lines = [ln.strip() for ln in output.strip().splitlines()]
    assert "bars 1-4: | C | F | x2" in lines


def test_interleave_compact_literal_bars(capsys):
    """Non-repeating bars are spelled out with merged labels."""
    from label_info import _print_interleave_compact

    tracks = {
        "chords": [
            LabelEntry(1.0, 3.0, "Am"),
            LabelEntry(3.0, 5.0, "G"),
        ],
        "parts": [
            LabelEntry(1.0, 5.0, "bridge"),
        ],
    }
    _print_interleave_compact(
        tracks, BAR_GRID_LARGE, BEAT_GRID_LARGE,
    )
    output = capsys.readouterr().out
    lines = [ln.strip() for ln in output.strip().splitlines()]
    assert "bar 1: Am bridge" in lines
    assert "bar 2: G" in lines


def test_interleave_compact_repeat_then_literal(capsys):
    """Repeat followed by non-repeating bars with section annotation."""
    from label_info import _print_interleave_compact

    # C F | C F | Am G  (repeat x2 then literals)
    tracks = {
        "chords": [
            LabelEntry(1.0, 3.0, "C"),
            LabelEntry(3.0, 5.0, "F"),
            LabelEntry(5.0, 7.0, "C"),
            LabelEntry(7.0, 9.0, "F"),
            LabelEntry(9.0, 11.0, "Am"),
            LabelEntry(11.0, 13.0, "G"),
        ],
        "parts": [
            LabelEntry(1.0, 9.0, "verse"),
            LabelEntry(9.0, 13.0, "bridge"),
        ],
    }
    _print_interleave_compact(
        tracks, BAR_GRID_LARGE, BEAT_GRID_LARGE,
    )
    output = capsys.readouterr().out
    lines = [ln.strip() for ln in output.strip().splitlines()]
    assert "bars 1-4: | C | F | x2  [verse]" in lines
    assert "bar 5: Am bridge" in lines
    assert "bar 6: G" in lines


def test_interleave_compact_primary_is_densest(capsys):
    """Primary track is the one with the most labels."""
    from label_info import _primary_track

    tracks = {
        "parts": [LabelEntry(1.0, 5.0, "verse")],
        "chords": [
            LabelEntry(1.0, 3.0, "C"),
            LabelEntry(3.0, 5.0, "F"),
        ],
    }
    name, labels = _primary_track(tracks)
    assert name == "chords"
    assert len(labels) == 2
