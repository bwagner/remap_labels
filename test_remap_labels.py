#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pytest",
#     "librosa>=0.11",
#     "soundfile",
#     "numpy",
# ]
# ///
"""Tests for v6 musical-structure label remapping (Fraction-based)."""

import os
from fractions import Fraction

import pytest

from remap_labels import LabelEntry, SectionEntry


# -- Fixtures --

BEATS_PER_BAR = 4

# Beat grid: 4 beats/bar, 0.5s per beat, starting at 1.0s
BEAT_GRID = [1.0 + i * 0.5 for i in range(32)]
BAR_GRID = [BEAT_GRID[i] for i in range(0, 32, 4)]

# Chords: C-F-Bb-F pattern, each 2 beats = 1/2 bar
CHORDS_SIMPLE = [
    LabelEntry(1.0, 2.0, "C"),
    LabelEntry(2.0, 3.0, "F"),
    LabelEntry(3.0, 4.0, "Bb"),
    LabelEntry(4.0, 5.0, "F"),
]

CHORDS_BRIDGE = [
    LabelEntry(5.0, 7.0, "Am"),
    LabelEntry(7.0, 9.0, "F"),
]

CHORDS_WITH_POINTS = [
    LabelEntry(1.0, 2.0, "C"),
    LabelEntry(3.5, 3.5, "C (piano)"),
    LabelEntry(5.0, 5.0, "C (piano)"),
]

PARTS_TWO_SECTIONS = [
    LabelEntry(1.0, 5.0, "verse"),
    LabelEntry(5.0, 9.0, "bridge"),
]

REGRESSION_DIR = os.path.join(os.path.dirname(__file__), "tests", "regression")
V6_BASELINE_DIR = os.path.join(REGRESSION_DIR, "v6_baseline")
GRIDS_DIR = os.path.join(REGRESSION_DIR, "grids")


# -- Tests for reconstruct_section --


class TestReconstructSection:

    def test_simple_reconstruction(self):
        from remap_labels import reconstruct_section

        entries = [
            SectionEntry(Fraction(0), "C", False, Fraction(1, 2)),
            SectionEntry(Fraction(1, 2), "F", False, Fraction(1, 2)),
            SectionEntry(Fraction(1), "Bb", False, Fraction(1, 2)),
            SectionEntry(Fraction(3, 2), "F", False, Fraction(1, 2)),
        ]
        new_beats = [2.0 + i * 0.6 for i in range(32)]
        new_bars = [new_beats[i] for i in range(0, 32, 4)]
        labels, warnings = reconstruct_section(
            entries, new_beats, new_bars, BEATS_PER_BAR,
            section_start_bar=0, section_end_bar=2,
        )
        assert len(labels) == 4
        assert labels[0] == LabelEntry(2.0, 3.2, "C")
        assert labels[1] == LabelEntry(3.2, 4.4, "F")
        assert labels[2] == LabelEntry(4.4, 5.6, "Bb")
        assert labels[3] == LabelEntry(5.6, 6.8, "F")

    def test_point_label(self):
        from remap_labels import reconstruct_section

        entries = [
            SectionEntry(Fraction(5, 4), "hit", True, Fraction(0)),
        ]
        labels, _ = reconstruct_section(
            entries, BEAT_GRID, BAR_GRID, BEATS_PER_BAR,
            section_start_bar=0, section_end_bar=4,
        )
        assert len(labels) == 1
        assert labels[0].is_point
        assert labels[0].start == BEAT_GRID[5]

    def test_section_truncation_clamps(self):
        from remap_labels import reconstruct_section

        entries = [
            SectionEntry(Fraction(0), "C", False, Fraction(1)),
            SectionEntry(Fraction(1), "F", False, Fraction(1)),
        ]
        # section_end_bar=1 means only 1 bar available, need 2
        labels, warnings = reconstruct_section(
            entries, BEAT_GRID, BAR_GRID, BEATS_PER_BAR,
            section_start_bar=0, section_end_bar=1,
        )
        assert len(warnings) > 0
        assert any("truncated" in w.lower() or "dropped" in w.lower() for w in warnings)

    def test_chained_output(self):
        from remap_labels import reconstruct_section

        entries = [
            SectionEntry(Fraction(0), "C", False, Fraction(1, 2)),
            SectionEntry(Fraction(1, 2), "F", False, Fraction(1, 2)),
            SectionEntry(Fraction(1), "Bb", False, Fraction(1, 2)),
        ]
        labels, _ = reconstruct_section(
            entries, BEAT_GRID, BAR_GRID, BEATS_PER_BAR,
            section_start_bar=0, section_end_bar=4,
        )
        for i in range(len(labels) - 1):
            if not labels[i].is_point and not labels[i + 1].is_point:
                assert labels[i].end == labels[i + 1].start


# -- Tests for validation --


class TestValidation:

    def test_matching_beat_counts(self):
        from remap_labels import validate_bar_beats

        warnings = validate_bar_beats(BEAT_GRID, BAR_GRID, BEAT_GRID, BAR_GRID)
        assert len(warnings) == 0

    def test_mismatched_beat_count(self):
        from remap_labels import validate_bar_beats

        new_beats = list(BEAT_GRID)
        new_beats.insert(5, 3.25)
        warnings = validate_bar_beats(BEAT_GRID, BAR_GRID, new_beats, BAR_GRID)
        assert len(warnings) > 0

    def test_different_bar_count(self):
        from remap_labels import validate_bar_beats

        fewer_bars = BAR_GRID[:3]
        warnings = validate_bar_beats(BEAT_GRID, BAR_GRID, BEAT_GRID, fewer_bars)
        assert any("bar count" in w.lower() for w in warnings)


# -- Tests against real song output --


def _load_grid(path):
    result = []
    from pathlib import Path

    for line in Path(path).read_text().splitlines():
        parts = line.strip().split()
        if parts:
            try:
                result.append(float(parts[0]))
            except ValueError:
                pass
    return result


def _load_chords(path):
    result = []
    from pathlib import Path

    for line in Path(path).read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            result.append((float(parts[0]), float(parts[1]), parts[2]))
    return result


def _bars_for_range(bars, start_time, end_time):
    """Return bar numbers (1-indexed) that fall within a time range."""
    result = []
    for i, t in enumerate(bars):
        if start_time - 0.01 <= t <= end_time + 0.01:
            result.append(i + 1)
    return result


class TestReviewMarks:
    """Review track should flag all reconstruction issues."""

    def test_dropped_chord_generates_review_mark(self):
        """When a chord is dropped, it appears in the review marks."""
        from remap_labels import reconstruct_section

        entries = [
            SectionEntry(Fraction(0), "C", False, Fraction(1, 2)),
            SectionEntry(Fraction(1, 2), "F", False, Fraction(1, 2)),
            SectionEntry(Fraction(1), "Bb", False, Fraction(1, 2)),  # beyond section
        ]
        # Only 1 bar available
        labels, warnings = reconstruct_section(
            entries, BEAT_GRID, BAR_GRID, BEATS_PER_BAR,
            section_start_bar=0, section_end_bar=1,
        )
        assert any("dropped" in w.lower() for w in warnings)

    def test_truncated_chord_generates_review_mark(self):
        """When a chord is truncated, it appears in the review marks."""
        from remap_labels import reconstruct_section

        entries = [
            SectionEntry(Fraction(0), "C", False, Fraction(2)),  # wants 2 bars
        ]
        # Only 1 bar available
        labels, warnings = reconstruct_section(
            entries, BEAT_GRID, BAR_GRID, BEATS_PER_BAR,
            section_start_bar=0, section_end_bar=1,
        )
        assert any("truncated" in w.lower() for w in warnings)

    def test_empty_bars_detected(self):
        """When a section has fewer chords than bars, flag empty bars."""
        from remap_labels import reconstruct_section

        # 1 bar of chords, but 3 bars of space
        entries = [
            SectionEntry(Fraction(0), "C", False, Fraction(1, 2)),
            SectionEntry(Fraction(1, 2), "F", False, Fraction(1, 2)),
        ]
        labels, warnings = reconstruct_section(
            entries, BEAT_GRID, BAR_GRID, BEATS_PER_BAR,
            section_start_bar=0, section_end_bar=3,
        )
        assert len(labels) == 2
        assert any("empty" in w.lower() for w in warnings)


class TestParseToAbsoluteBarBeat:
    """v7: parse labels to absolute (bar, beat) without sections."""

    def test_simple_chords(self):
        from remap_labels import parse_labels_to_bar_beat

        entries = parse_labels_to_bar_beat(CHORDS_SIMPLE, BEAT_GRID, BAR_GRID)
        assert len(entries) == 4
        # C at bar 0 beat 0
        assert entries[0].bar_offset == Fraction(0)
        assert entries[0].label == "C"
        # F at bar 0 beat 2 = bar 1/2
        assert entries[1].bar_offset == Fraction(1, 2)
        # Bb at bar 1 beat 0
        assert entries[2].bar_offset == Fraction(1)
        # F at bar 1 beat 2 = bar 3/2
        assert entries[3].bar_offset == Fraction(3, 2)

    def test_bridge_chords(self):
        from remap_labels import parse_labels_to_bar_beat

        entries = parse_labels_to_bar_beat(CHORDS_BRIDGE, BEAT_GRID, BAR_GRID)
        assert len(entries) == 2
        # Am at bar 2
        assert entries[0].bar_offset == Fraction(2)
        assert entries[0].bar_count == Fraction(1)
        # F at bar 3
        assert entries[1].bar_offset == Fraction(3)

    def test_point_labels(self):
        from remap_labels import parse_labels_to_bar_beat

        entries = parse_labels_to_bar_beat(CHORDS_WITH_POINTS, BEAT_GRID, BAR_GRID)
        assert len(entries) == 3
        assert entries[1].is_point
        # Point at 3.5 = bar 1, beat 1 = bar 5/4
        assert entries[1].bar_offset == Fraction(5, 4)


    def test_sub_beat_labels_get_distinct_offsets(self):
        """Labels between beats must not collapse to the same bar_offset.

        Turnaround scenario: F7, Bb7, B7 all within one beat span.
        Each must get a distinct bar_offset so they reconstruct at
        different times.
        """
        from remap_labels import parse_labels_to_bar_beat

        # Bar 1 has beats at 1.0, 1.5, 2.0, 2.5
        # Place 3 chords between beat 1 (1.0) and beat 2 (1.5):
        #   F7 at 1.0 (on beat 1)
        #   Bb7 at 1.16 (between beat 1 and 2)
        #   B7 at 1.32 (between beat 1 and 2, closer to beat 2)
        turnaround = [
            LabelEntry(1.0, 1.16, "F7"),
            LabelEntry(1.16, 1.32, "Bb7"),
            LabelEntry(1.32, 1.5, "B7"),
        ]
        entries = parse_labels_to_bar_beat(turnaround, BEAT_GRID, BAR_GRID)

        offsets = [e.bar_offset for e in entries]
        assert len(set(offsets)) == 3, (
            f"Sub-beat labels collapsed to same offset: {offsets}"
        )

    def test_sub_beat_round_trip(self):
        """Sub-beat labels must reconstruct at distinct, ordered times."""
        from remap_labels import parse_labels_to_bar_beat, reconstruct_labels

        turnaround = [
            LabelEntry(1.0, 1.16, "F7"),
            LabelEntry(1.16, 1.32, "Bb7"),
            LabelEntry(1.32, 1.5, "B7"),
        ]
        entries = parse_labels_to_bar_beat(turnaround, BEAT_GRID, BAR_GRID)
        labels, _ = reconstruct_labels(
            entries, BEAT_GRID, BAR_GRID, BEATS_PER_BAR,
        )

        assert len(labels) == 3
        # All must have distinct start times, in order
        starts = [lbl.start for lbl in labels]
        assert starts[0] < starts[1] < starts[2], (
            f"Sub-beat labels not in order: {starts}"
        )


    def test_pickup_label_before_bar_1(self):
        """Labels before bar 1 (pickup/anacrusis) get negative bar_offset."""
        from remap_labels import parse_labels_to_bar_beat

        # Beat grid starts at 0.5 (pickup beat), bar 1 at 1.0
        # Label at 0.5 is one beat before bar 1
        beat_grid = [0.5 + i * 0.5 for i in range(32)]
        bar_grid = [beat_grid[1 + i * 4] for i in range(7)]  # bar 1 at 1.0

        pickup = [
            LabelEntry(0.5, 1.0, "Dbm7b5"),
        ]
        entries = parse_labels_to_bar_beat(pickup, beat_grid, bar_grid)

        assert len(entries) == 1
        assert entries[0].bar_offset < 0, (
            f"Pickup label should have negative bar_offset, got {entries[0].bar_offset}"
        )

    def test_pickup_reconstructs_before_bar_1(self):
        """Pickup labels reconstruct at times before bar 1."""
        from remap_labels import parse_labels_to_bar_beat, reconstruct_labels

        beat_grid = [0.5 + i * 0.5 for i in range(32)]
        bar_grid = [beat_grid[1 + i * 4] for i in range(7)]  # bar 1 at 1.0

        pickup = [
            LabelEntry(0.5, 1.0, "Dbm7b5"),
            LabelEntry(1.0, 3.0, "Am"),  # on bar 1
        ]
        entries = parse_labels_to_bar_beat(pickup, beat_grid, bar_grid)
        labels, _ = reconstruct_labels(
            entries, beat_grid, bar_grid, 4,
        )

        assert len(labels) == 2
        assert labels[0].start < bar_grid[0], (
            f"Pickup should be before bar 1 ({bar_grid[0]}), got {labels[0].start}"
        )
        assert labels[0].start < labels[1].start
        assert labels[0].end <= labels[1].start + 0.001

    def test_anacrusis_sub_beat_end_preserved(self):
        """Label ending mid-anacrusis-beat should not snap to bar 1."""
        from remap_labels import parse_labels_to_bar_beat, reconstruct_labels

        beat_grid = [0.5 + i * 0.5 for i in range(32)]
        bar_grid = [beat_grid[1 + i * 4] for i in range(7)]  # bar 1 at 1.0

        # Chord starts on anacrusis beat (0.5) and ends mid-anacrusis-beat (0.75)
        anacrusis_sub = [LabelEntry(0.5, 0.75, "Dbm7b5")]
        entries = parse_labels_to_bar_beat(anacrusis_sub, beat_grid, bar_grid)
        labels, _ = reconstruct_labels(entries, beat_grid, bar_grid, 4)

        assert len(labels) == 1
        assert abs(labels[0].end - 0.75) < 0.01, (
            f"Sub-beat anacrusis end should stay at 0.75, got {labels[0].end}"
        )

    def test_sub_beat_past_last_beat_in_bar_keeps_duration(self):
        """Label spanning bar-N beat-4 to past-beat-4 must not collapse to a point."""
        from remap_labels import parse_labels_to_bar_beat, reconstruct_labels

        # beat_grid 0.5s spacing, bar every 4 beats -> bar 0 beats at 0,0.5,1.0,1.5; bar 1 at 2.0
        beat_grid = [0.5 * i for i in range(32)]
        bar_grid = [beat_grid[i] for i in range(0, 32, 4)]

        # Chord: bar 0 beat 4 (1.5) to mid-way between beat 4 and bar 1 (1.85)
        chord = [
            LabelEntry(1.5, 1.85, "X"),
            LabelEntry(1.85, 2.0, "Y"),
        ]
        entries = parse_labels_to_bar_beat(chord, beat_grid, bar_grid)
        labels, _ = reconstruct_labels(entries, beat_grid, bar_grid, 4)

        assert len(labels) == 2
        assert labels[0].end > labels[0].start, (
            f"X must have duration, got start={labels[0].start} end={labels[0].end}"
        )
        assert abs(labels[0].end - 1.85) < 0.05, (
            f"X end should be near 1.85 (mid-way past beat 4), got {labels[0].end}"
        )
        assert labels[0].end <= labels[1].start + 0.001, "X-Y must not overlap"

    def test_anacrusis_chain_no_overlap(self):
        """Chained sub-beat chords in anacrusis must stay chained after round-trip."""
        from remap_labels import parse_labels_to_bar_beat, reconstruct_labels

        beat_grid = [0.5 + i * 0.5 for i in range(32)]
        bar_grid = [beat_grid[1 + i * 4] for i in range(7)]  # bar 1 at 1.0

        chain = [
            LabelEntry(0.5, 0.75, "Dbm7b5"),        # first half of anacrusis beat
            LabelEntry(0.75, 1.0, "Dbm7b5/E"),      # second half of anacrusis beat
            LabelEntry(1.0, 3.0, "Am"),             # bar 1 onwards
        ]
        entries = parse_labels_to_bar_beat(chain, beat_grid, bar_grid)
        labels, _ = reconstruct_labels(entries, beat_grid, bar_grid, 4)

        assert len(labels) == 3
        for i in range(len(labels) - 1):
            assert labels[i].end <= labels[i + 1].start + 0.001, (
                f"Overlap: '{labels[i].label}' ends at {labels[i].end:.6f} "
                f"but '{labels[i+1].label}' starts at {labels[i+1].start:.6f}"
            )

    def test_non_overlapping_input_produces_non_overlapping_output(self):
        """If input labels don't overlap, output labels must not overlap either.

        Sub-beat labels (like turnarounds) must not produce overlapping
        times after reconstruction.
        """
        from remap_labels import parse_labels_to_bar_beat, reconstruct_labels

        # Turnaround: 4 chords packed within one beat, non-overlapping
        turnaround = [
            LabelEntry(1.0, 1.16, "F7"),
            LabelEntry(1.16, 1.32, "Bb7"),
            LabelEntry(1.32, 1.5, "B7"),
            LabelEntry(1.5, 3.0, "C7"),
        ]
        # Verify input doesn't overlap
        for i in range(len(turnaround) - 1):
            assert turnaround[i].end <= turnaround[i + 1].start + 0.001

        entries = parse_labels_to_bar_beat(turnaround, BEAT_GRID, BAR_GRID)
        labels, _ = reconstruct_labels(
            entries, BEAT_GRID, BAR_GRID, BEATS_PER_BAR,
        )

        assert len(labels) == 4
        for i in range(len(labels) - 1):
            assert labels[i].end <= labels[i + 1].start + 0.001, (
                f"Overlap: '{labels[i].label}' ends at {labels[i].end:.6f} "
                f"but '{labels[i+1].label}' starts at {labels[i+1].start:.6f}"
            )


class TestReconstructAbsolute:
    """v7: reconstruct from absolute positions without sections."""

    def test_simple_reconstruction(self):
        from remap_labels import parse_labels_to_bar_beat, reconstruct_labels

        entries = parse_labels_to_bar_beat(CHORDS_SIMPLE, BEAT_GRID, BAR_GRID)
        # Reconstruct on a grid with different spacing
        new_beats = [2.0 + i * 0.6 for i in range(32)]
        new_bars = [new_beats[i] for i in range(0, 32, 4)]
        labels, warnings = reconstruct_labels(
            entries, new_beats, new_bars, BEATS_PER_BAR,
        )
        assert len(labels) == 4
        assert labels[0] == LabelEntry(2.0, 3.2, "C")
        assert labels[1] == LabelEntry(3.2, 4.4, "F")

    def test_chained_output(self):
        from remap_labels import parse_labels_to_bar_beat, reconstruct_labels

        all_chords = CHORDS_SIMPLE + CHORDS_BRIDGE
        entries = parse_labels_to_bar_beat(all_chords, BEAT_GRID, BAR_GRID)
        labels, _ = reconstruct_labels(entries, BEAT_GRID, BAR_GRID, BEATS_PER_BAR)
        # Range labels should chain
        for i in range(len(labels) - 1):
            if not labels[i].is_point and not labels[i + 1].is_point:
                assert labels[i].end == labels[i + 1].start

    def test_dropped_label_beyond_grid(self):
        from remap_labels import reconstruct_labels

        entries = [
            SectionEntry(Fraction(10), "C", False, Fraction(1, 2)),  # bar 10
        ]
        # Grid with only 4 bars
        short_beats = [i * 0.5 for i in range(16)]
        short_bars = [i * 2.0 for i in range(4)]
        labels, warnings = reconstruct_labels(
            entries, short_beats, short_bars, BEATS_PER_BAR,
        )
        assert len(labels) == 0
        assert any("dropped" in w.lower() for w in warnings)


class TestBaselineIntegrity:
    """Verify the checked-in baseline has expected properties."""

    def test_baseline_has_expected_files(self):
        """Baseline directory contains chords, parts, guit label files."""
        from pathlib import Path
        baseline = Path(V6_BASELINE_DIR)
        if not baseline.exists():
            pytest.skip("Baseline directory not found")
        names = {f.name for f in baseline.glob("*.txt")}
        assert "chords_blues_brothers_everybody_needs_somebody.txt" in names
        assert "parts_blues_brothers_everybody_needs_somebody.txt" in names
        assert "guit_blues_brothers_everybody_needs_somebody.txt" in names

    def test_review_contains_structural_warnings(self):
        """Review track should flag structural changes."""
        from pathlib import Path

        review_path = Path(V6_BASELINE_DIR) / "review.txt"
        if not review_path.exists():
            pytest.skip("Baseline review.txt not found")

        content = review_path.read_text()
        assert "old section missing" in content.lower(), (
            "Review track should flag removed sections"
        )


REAL_DATA_AVAILABLE = (
    os.path.exists(f"{V6_BASELINE_DIR}/chords_blues_brothers_everybody_needs_somebody.txt")
    and os.path.exists(f"{GRIDS_DIR}/new_bars.txt")
    and os.path.exists(f"{GRIDS_DIR}/new_beats.txt")
)


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="Run remap_labels on real data first")
class TestRealSongOutput:

    def test_all_chord_boundaries_on_beats(self):
        chords = _load_chords(
            f"{V6_BASELINE_DIR}/chords_blues_brothers_everybody_needs_somebody.txt"
        )
        beats = _load_grid(f"{GRIDS_DIR}/new_beats.txt")

        off_beat = []
        for start, end, label in chords:
            if abs(start - end) < 0.001:
                continue
            for t, which in [(start, "start"), (end, "end")]:
                if not any(abs(t - b) < 0.01 for b in beats):
                    off_beat.append(f"{label} {which} at {t:.3f} not on any beat")

        assert off_beat == [], (
            f"{len(off_beat)} boundaries off-beat:\n" + "\n".join(off_beat[:20])
        )

    def test_bars_31_32_not_empty(self):
        """Bars 31-32 must contain chords."""
        chords = _load_chords(
            f"{V6_BASELINE_DIR}/chords_blues_brothers_everybody_needs_somebody.txt"
        )
        bars = _load_grid(f"{GRIDS_DIR}/new_bars.txt")

        bar_31_start = bars[30]
        bar_32_end = bars[32] if len(bars) > 32 else bars[-1]

        chords_in_range = [
            (s, e, label) for s, e, label in chords
            if not (abs(s - e) < 0.001) and e > bar_31_start + 0.001 and s < bar_32_end - 0.001
        ]
        assert len(chords_in_range) > 0, (
            f"No chords in bars 31-32 ({bar_31_start:.3f}-{bar_32_end:.3f})"
        )

    def test_structural_change_at_bar_98_flagged(self):
        """The removed section around bar 98 must be flagged in the review track."""
        import os
        review_path = f"{V6_BASELINE_DIR}/review.txt"
        if not os.path.exists(review_path):
            pytest.skip("Run remap_labels on real data first")

        review = _load_chords(review_path)
        bars = _load_grid(f"{GRIDS_DIR}/new_bars.txt")

        bar_98_time = bars[97]
        nearby_marks = [
            label for s, e, label in review
            if abs(s - bar_98_time) < 5.0
        ]
        assert len(nearby_marks) > 0, (
            "Structural change near bar 98 not flagged in review track"
        )

    def test_hiiit_section_bar_crossing_flagged(self):
        """The held C chord in 'hiiit' crosses bar boundaries (structural
        change). This must be flagged in the review track."""
        import os
        review_path = f"{V6_BASELINE_DIR}/review.txt"
        if not os.path.exists(review_path):
            pytest.skip("Run remap_labels on real data first")

        review = _load_chords(review_path)
        bars = _load_grid(f"{GRIDS_DIR}/new_bars.txt")

        # The 'hiiit' section is near bar 155
        bar_155_time = bars[154]
        nearby_marks = [
            label for s, e, label in review
            if abs(s - bar_155_time) < 10.0
        ]
        assert len(nearby_marks) > 0, (
            "Structural change near hiiit section not flagged in review track"
        )


# -- Bar shift detection --

AUDIO_CLIPS_DIR = os.path.join(REGRESSION_DIR, "audio_clips")

AUDIO_CLIPS_AVAILABLE = os.path.exists(
    os.path.join(AUDIO_CLIPS_DIR, "shc_old_intro.mp3")
)


class TestChooseBarShift:
    """Test the shift selection logic independent of DTW."""

    def test_clear_majority(self):
        """When one shift dominates, pick it confidently."""
        from remap_labels import choose_bar_shift

        shift, candidates = choose_bar_shift([2, 2, 2, 2, 2, 2, 2, 1])
        assert shift == 2
        assert candidates[0]["shift"] == 2
        assert candidates[0]["confident"]

    def test_ambiguous_shifts(self):
        """When top two are close, flag as not confident."""
        from remap_labels import choose_bar_shift

        shift, candidates = choose_bar_shift([0, -1, 0, 0, -1, -1, -1, -1])
        assert len(candidates) >= 2
        assert not candidates[0]["confident"]

    def test_single_shift(self):
        """All bars agree - maximally confident."""
        from remap_labels import choose_bar_shift

        shift, candidates = choose_bar_shift([0, 0, 0, 0])
        assert shift == 0
        assert candidates[0]["confident"]
        assert len(candidates) == 1

    def test_three_way_split(self):
        """Three candidates, none dominant."""
        from remap_labels import choose_bar_shift

        shift, candidates = choose_bar_shift([0, 1, 2, 0, 1, 2])
        assert not candidates[0]["confident"]
        assert len(candidates) >= 3

    def test_clear_majority_with_outliers(self):
        """Strong majority with a few outliers is still confident."""
        from remap_labels import choose_bar_shift

        shift, candidates = choose_bar_shift([2, 2, 2, 2, 2, 2, 2, 2, 5, -3])
        assert shift == 2
        assert candidates[0]["confident"]

    def test_returns_all_candidates_sorted(self):
        """Candidates are sorted by vote count descending."""
        from remap_labels import choose_bar_shift

        shift, candidates = choose_bar_shift([0, 0, 0, -1, -1, 1])
        votes = [c["votes"] for c in candidates]
        assert votes == sorted(votes, reverse=True)

    def test_confidence_at_exact_threshold(self):
        """At exactly 2:1 ratio, still confident."""
        from remap_labels import choose_bar_shift

        shift, candidates = choose_bar_shift([0, 0, 0, 0, -1, -1])
        assert shift == 0
        assert candidates[0]["confident"]

    def test_confidence_just_below_threshold(self):
        """Just below 2:1 ratio, not confident."""
        from remap_labels import choose_bar_shift

        shift, candidates = choose_bar_shift([0, 0, 0, -1, -1])
        assert shift == 0
        assert not candidates[0]["confident"]

    def test_candidates_include_bars(self):
        """Each candidate lists which bars voted for it."""
        from remap_labels import choose_bar_shift

        shift, candidates = choose_bar_shift([2, 1, 2, 2])
        top = candidates[0]
        assert top["shift"] == 2
        assert top["bars"] == [1, 3, 4]  # 1-indexed


@pytest.mark.skipif(not AUDIO_CLIPS_AVAILABLE, reason="Audio clips not available")
class TestBarShiftDetection:
    """Bar shift must use majority vote, not just bar 1."""

    def test_bar_shift_majority_vote(self):
        """Sweet Home Chicago: old has 2-bar intro riff, new has 4-bar.

        DTW matches bar 1 to the wrong repetition (shift=+1),
        but bars 2+ consistently give shift=+2. Majority vote should
        detect the correct shift of +2.
        """
        from remap_labels import (
            compute_alignment,
            determine_bar_shift,
            load_timestamps,
            make_warp_func,
        )

        old_bars = load_timestamps(f"{AUDIO_CLIPS_DIR}/shc_old_bars.txt")
        new_bars = load_timestamps(f"{AUDIO_CLIPS_DIR}/shc_new_bars.txt")

        old_times, new_times = compute_alignment(
            f"{AUDIO_CLIPS_DIR}/shc_old_intro.mp3",
            f"{AUDIO_CLIPS_DIR}/shc_new_intro.mp3",
        )
        warp = make_warp_func(old_times, new_times)

        shift = determine_bar_shift(old_bars, new_bars, warp)
        assert shift == 2, (
            f"Expected bar shift +2, got +{shift}. "
            "DTW likely matched intro riff to wrong repetition."
        )

    def test_bar_shift_when_old_longer_than_new(self):
        """When old audio is much longer than new, subsequence DTW
        should find the new content within the old.

        Uses SHC clips reversed: 27s new clip as 'old', 11s old clip
        as 'new', simulating a partial-song scenario like Minnie.
        """
        from remap_labels import (
            compute_alignment,
            determine_bar_shift,
            load_timestamps,
            make_warp_func,
        )

        # Reversed: long clip as "old", short clip as "new"
        long_bars = load_timestamps(f"{AUDIO_CLIPS_DIR}/shc_new_bars.txt")
        short_bars = load_timestamps(f"{AUDIO_CLIPS_DIR}/shc_old_short_bars.txt")

        old_times, new_times = compute_alignment(
            f"{AUDIO_CLIPS_DIR}/shc_new_intro.mp3",
            f"{AUDIO_CLIPS_DIR}/shc_old_short.mp3",
        )
        warp = make_warp_func(old_times, new_times)

        shift = determine_bar_shift(long_bars, short_bars, warp)
        # The short clip's content starts at bar 1 of the long clip
        # (after 2 extra intro bars), so shift should be negative
        assert shift <= 0, (
            f"Expected non-positive bar shift, got +{shift}. "
            "Subsequence DTW should find short clip within long clip."
        )


# -- Tests for dir-mode CLI discovery --


class TestDiscoverDirInputs:
    """discover_dir_inputs(directory, want_labels) picks audio + beats/bars/labels by stem."""

    def _make(self, tmp_path, files):
        for name in files:
            (tmp_path / name).write_text("")
        return tmp_path

    def test_single_audio_gets_stem(self, tmp_path):
        from remap_labels import discover_dir_inputs

        d = self._make(tmp_path, [
            "song_a.mp3",
            "beats_song_a.txt",
            "bars_song_a.txt",
        ])
        result = discover_dir_inputs(d, want_labels=False)
        assert result.audio.name == "song_a.mp3"
        assert result.beats.name == "beats_song_a.txt"
        assert result.bars.name == "bars_song_a.txt"

    def test_zero_audio_errors(self, tmp_path):
        from remap_labels import discover_dir_inputs

        d = self._make(tmp_path, ["beats_x.txt", "bars_x.txt"])
        with pytest.raises(ValueError, match="no audio"):
            discover_dir_inputs(d, want_labels=False)

    def test_multiple_audio_errors(self, tmp_path):
        from remap_labels import discover_dir_inputs

        d = self._make(tmp_path, [
            "a.mp3", "b.opus",
            "beats_a.txt", "bars_a.txt",
        ])
        with pytest.raises(ValueError, match="multiple audio"):
            discover_dir_inputs(d, want_labels=False)

    def test_missing_beats_errors(self, tmp_path):
        from remap_labels import discover_dir_inputs

        d = self._make(tmp_path, ["song.mp3", "bars_song.txt"])
        with pytest.raises(ValueError, match="beats_song.txt"):
            discover_dir_inputs(d, want_labels=False)

    def test_missing_bars_errors(self, tmp_path):
        from remap_labels import discover_dir_inputs

        d = self._make(tmp_path, ["song.mp3", "beats_song.txt"])
        with pytest.raises(ValueError, match="bars_song.txt"):
            discover_dir_inputs(d, want_labels=False)

    def test_finds_labels_matching_stem(self, tmp_path):
        from remap_labels import discover_dir_inputs

        d = self._make(tmp_path, [
            "song.mp3",
            "beats_song.txt",
            "bars_song.txt",
            "chords_song.txt",
            "parts_song.txt",
            "guit_song.txt",
        ])
        result = discover_dir_inputs(d, want_labels=True)
        label_names = sorted(p.name for p in result.labels)
        assert label_names == ["chords_song.txt", "guit_song.txt", "parts_song.txt"]

    def test_ignores_unrelated_stem_files(self, tmp_path):
        from remap_labels import discover_dir_inputs

        d = self._make(tmp_path, [
            "song.mp3",
            "beats_song.txt",
            "bars_song.txt",
            "chords_song.txt",
            "chords_other_song.txt",  # different stem, ignored
            "notes.txt",               # no stem suffix, ignored
        ])
        result = discover_dir_inputs(d, want_labels=True)
        label_names = [p.name for p in result.labels]
        assert label_names == ["chords_song.txt"]

    def test_new_dir_skips_labels(self, tmp_path):
        from remap_labels import discover_dir_inputs

        d = self._make(tmp_path, [
            "new_song.opus",
            "beats_new_song.txt",
            "bars_new_song.txt",
            "chords_new_song.txt",  # present but not requested
        ])
        result = discover_dir_inputs(d, want_labels=False)
        assert result.labels is None

    def test_all_audio_exts_recognized(self, tmp_path):
        from remap_labels import discover_dir_inputs

        for ext in ("mp3", "m4a", "opus", "wav", "flac", "ogg", "aac", "wma"):
            sub = tmp_path / ext
            sub.mkdir()
            self._make(sub, [f"x.{ext}", "beats_x.txt", "bars_x.txt"])
            result = discover_dir_inputs(sub, want_labels=False)
            assert result.audio.suffix == f".{ext}"


# -- Tests for subseq mode resolution --


class TestResolveSubseqMode:
    """resolve_subseq_mode turns 'auto'/'on'/'off' into the DTW subseq flag."""

    def test_on_forces_true_regardless_of_bars(self):
        from remap_labels import resolve_subseq_mode

        assert resolve_subseq_mode("on", 72, 72) is True
        assert resolve_subseq_mode("on", 10, 72) is True

    def test_off_forces_false_regardless_of_bars(self):
        from remap_labels import resolve_subseq_mode

        assert resolve_subseq_mode("off", 72, 72) is False
        assert resolve_subseq_mode("off", 10, 72) is False

    def test_auto_equal_bars_off(self):
        from remap_labels import resolve_subseq_mode

        assert resolve_subseq_mode("auto", 72, 72) is False

    def test_auto_close_bars_off(self):
        from remap_labels import resolve_subseq_mode

        # 60/72 = 0.83, well above threshold 0.5
        assert resolve_subseq_mode("auto", 60, 72) is False

    def test_auto_very_different_bars_on(self):
        from remap_labels import resolve_subseq_mode

        # 10/72 = 0.14, below threshold 0.5
        assert resolve_subseq_mode("auto", 10, 72) is True

    def test_auto_at_boundary_stays_off(self):
        from remap_labels import resolve_subseq_mode

        # 36/72 = 0.5 exactly, NOT below threshold -> off
        assert resolve_subseq_mode("auto", 36, 72) is False

    def test_auto_just_below_boundary_on(self):
        from remap_labels import resolve_subseq_mode

        # 35/72 = 0.486, below 0.5 -> on
        assert resolve_subseq_mode("auto", 35, 72) is True

    def test_invalid_mode_raises(self):
        from remap_labels import resolve_subseq_mode

        with pytest.raises(ValueError):
            resolve_subseq_mode("maybe", 72, 72)
