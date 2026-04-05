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

SONG_DIR = (
    "/Users/bwagner/projects/cover-notes-gitlab/batch01"
    "/blues_brothers_everybody_needs_somebody"
)
REMAPPED_DIR = "/Users/bwagner/projects/remap_labels/remapped"


# -- Tests for parse_section_entries --


class TestParseSectionEntries:

    def test_half_bar_chords(self):
        from remap_labels import parse_section_entries

        entries = parse_section_entries(
            CHORDS_SIMPLE, BEAT_GRID, BAR_GRID, BEATS_PER_BAR, 1.0, 5.0,
        )
        assert len(entries) == 4
        assert entries[0] == SectionEntry(Fraction(0), "C", False, Fraction(1, 2))
        assert entries[1] == SectionEntry(Fraction(1, 2), "F", False, Fraction(1, 2))
        assert entries[2] == SectionEntry(Fraction(1), "Bb", False, Fraction(1, 2))
        assert entries[3] == SectionEntry(Fraction(3, 2), "F", False, Fraction(1, 2))

    def test_full_bar_chords(self):
        from remap_labels import parse_section_entries

        entries = parse_section_entries(
            CHORDS_BRIDGE, BEAT_GRID, BAR_GRID, BEATS_PER_BAR, 5.0, 9.0,
        )
        assert len(entries) == 2
        assert entries[0] == SectionEntry(Fraction(0), "Am", False, Fraction(1))
        assert entries[1] == SectionEntry(Fraction(1), "F", False, Fraction(1))

    def test_point_labels(self):
        from remap_labels import parse_section_entries

        entries = parse_section_entries(
            CHORDS_WITH_POINTS, BEAT_GRID, BAR_GRID, BEATS_PER_BAR, 1.0, 7.0,
        )
        assert len(entries) == 3
        assert entries[0] == SectionEntry(Fraction(0), "C", False, Fraction(1, 2))
        assert entries[1] == SectionEntry(Fraction(5, 4), "C (piano)", True, Fraction(0))
        assert entries[2] == SectionEntry(Fraction(2), "C (piano)", True, Fraction(0))

    def test_filters_chords_outside_section(self):
        from remap_labels import parse_section_entries

        all_chords = CHORDS_SIMPLE + CHORDS_BRIDGE
        entries = parse_section_entries(
            all_chords, BEAT_GRID, BAR_GRID, BEATS_PER_BAR, 1.0, 5.0,
        )
        assert len(entries) == 4


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


# -- Tests for map_section_boundaries --


class TestMapSectionBoundaries:

    def test_identity_warp(self):
        from remap_labels import map_section_boundaries

        def identity_warp(t):
            return t

        result = map_section_boundaries(PARTS_TWO_SECTIONS, identity_warp, BAR_GRID)
        assert result[0].name == "verse"
        assert result[0].new_bar_idx == 0
        assert result[1].name == "bridge"
        assert result[1].new_bar_idx == 2

    def test_shifted_warp(self):
        from remap_labels import map_section_boundaries

        def shift_warp(t):
            return t + 1.0

        new_bars = [2.0 + i * 2.0 for i in range(8)]
        result = map_section_boundaries(PARTS_TWO_SECTIONS, shift_warp, new_bars)
        assert result[0].new_bar_idx == 0
        assert result[1].new_bar_idx == 2


# -- Tests for non-chord labels --


class TestNonChordLabels:

    def test_guit_label_in_section(self):
        from remap_labels import parse_section_entries

        guit_labels = [LabelEntry(2.5, 3.0, "gB eB -> gC eC")]
        entries = parse_section_entries(
            guit_labels, BEAT_GRID, BAR_GRID, BEATS_PER_BAR, 1.0, 5.0,
        )
        assert len(entries) == 1
        assert entries[0].bar_offset == Fraction(3, 4)
        assert entries[0].bar_count == Fraction(1, 4)


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


class TestNoGapsBetweenSections:
    """When DTW maps sections with a gap, chords must still be contiguous."""

    def test_section_gap_filled(self):
        """If old section A has 2 bars of chords and DTW would map next
        section to bar 4 (leaving bars 2-3 empty), contiguous placement
        should close the gap."""
        from remap_labels import (
            map_section_boundaries,
            parse_section_entries,
            reconstruct_section,
        )

        # Old grids: 4 beats/bar, bars at 0, 2, 4, 6, 8, 10
        old_beats = [i * 0.5 for i in range(24)]
        old_bars = [i * 2.0 for i in range(6)]

        # Section A: bars 0-1 (0.0 - 4.0), 4 half-bar chords
        section_a_chords = [
            LabelEntry(0.0, 1.0, "C"),
            LabelEntry(1.0, 2.0, "F"),
            LabelEntry(2.0, 3.0, "Bb"),
            LabelEntry(3.0, 4.0, "F"),
        ]

        # Section B: bars 2-3 (4.0 - 8.0), 4 half-bar chords
        section_b_chords = [
            LabelEntry(4.0, 5.0, "Am"),
            LabelEntry(5.0, 6.0, "F"),
            LabelEntry(6.0, 7.0, "Am"),
            LabelEntry(7.0, 8.0, "G"),
        ]

        old_parts = [
            LabelEntry(0.0, 4.0, "verse"),
            LabelEntry(4.0, 8.0, "bridge"),
        ]

        # New grids: more bars (DTW would put bridge at bar 4)
        new_beats = [i * 0.5 for i in range(32)]
        new_bars = [i * 2.0 for i in range(8)]

        # Warp that shifts section B later (simulating DTW imprecision)
        def warp(t):
            if t >= 4.0:
                return t + 4.0  # shifts bridge to 8.0 -> bar 4
            return t

        # Map with old_bar_grid -> contiguous placement
        mapped = map_section_boundaries(old_parts, warp, new_bars, old_bars)

        # Section A at bar 0, section B should be at bar 2 (contiguous),
        # NOT bar 4 (where DTW would put it)
        assert mapped[0].new_bar_idx == 0
        assert mapped[1].new_bar_idx == 2, (
            f"Bridge should be at bar 2 (contiguous), got bar {mapped[1].new_bar_idx}"
        )

        # Reconstruct and verify no gap
        entries_a = parse_section_entries(
            section_a_chords, old_beats, old_bars, BEATS_PER_BAR, 0.0, 4.0,
        )
        entries_b = parse_section_entries(
            section_b_chords, old_beats, old_bars, BEATS_PER_BAR, 4.0, 8.0,
        )

        labels_a, _ = reconstruct_section(
            entries_a, new_beats, new_bars, BEATS_PER_BAR,
            section_start_bar=mapped[0].new_bar_idx,
            section_end_bar=mapped[1].new_bar_idx,
        )
        labels_b, _ = reconstruct_section(
            entries_b, new_beats, new_bars, BEATS_PER_BAR,
            section_start_bar=mapped[1].new_bar_idx,
            section_end_bar=mapped[1].new_bar_idx + 2,
        )

        # Last chord of A should end where first chord of B starts
        assert labels_a[-1].end == labels_b[0].start, (
            f"Gap between sections: A ends {labels_a[-1].end}, "
            f"B starts {labels_b[0].start}"
        )


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


REAL_DATA_AVAILABLE = (
    os.path.exists(f"{REMAPPED_DIR}/chords_blues_brothers_everybody_needs_somebody.txt")
    and os.path.exists(f"{SONG_DIR}/new_bars.txt")
    and os.path.exists(f"{SONG_DIR}/new_beats.txt")
)


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="Run remap_labels on real data first")
class TestRealSongOutput:

    def test_all_chord_boundaries_on_beats(self):
        chords = _load_chords(
            f"{REMAPPED_DIR}/chords_blues_brothers_everybody_needs_somebody.txt"
        )
        beats = _load_grid(f"{SONG_DIR}/new_beats.txt")

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
            f"{REMAPPED_DIR}/chords_blues_brothers_everybody_needs_somebody.txt"
        )
        bars = _load_grid(f"{SONG_DIR}/new_bars.txt")

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
        review_path = f"{REMAPPED_DIR}/review.txt"
        if not os.path.exists(review_path):
            pytest.skip("Run remap_labels on real data first")

        review = _load_chords(review_path)
        bars = _load_grid(f"{SONG_DIR}/new_bars.txt")

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
        review_path = f"{REMAPPED_DIR}/review.txt"
        if not os.path.exists(review_path):
            pytest.skip("Run remap_labels on real data first")

        review = _load_chords(review_path)
        bars = _load_grid(f"{SONG_DIR}/new_bars.txt")

        # The 'hiiit' section is near bar 155
        bar_155_time = bars[154]
        nearby_marks = [
            label for s, e, label in review
            if abs(s - bar_155_time) < 10.0
        ]
        assert len(nearby_marks) > 0, (
            "Structural change near hiiit section not flagged in review track"
        )
