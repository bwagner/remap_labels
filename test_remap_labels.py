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
V6_BASELINE_DIR = "/Users/bwagner/projects/remap_labels/tests/regression/v6_baseline"


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


class TestV7OutputMatchesV6:
    """Removing section-based logic must not change the label output."""

    def _load_all_labels(self, directory):
        """Load all label files from a directory, return dict of filename -> content."""
        from pathlib import Path
        result = {}
        for f in sorted(Path(directory).glob("*.txt")):
            if f.name == "review.txt":
                continue  # review wording will change
            result[f.name] = f.read_text()
        return result

    def test_label_output_unchanged(self):
        """Label files (chords, parts, guit) from v7 must match v6."""
        import os
        v6_dir = V6_BASELINE_DIR
        v7_dir = "/Users/bwagner/projects/remap_labels/remapped_v7"
        if not os.path.exists(f"{v6_dir}/chords_blues_brothers_everybody_needs_somebody.txt"):
            pytest.skip("Run v6 remap_labels on real data first")
        if not os.path.exists(v7_dir):
            pytest.skip("Run v7 remap_labels on real data first")

        v6_labels = self._load_all_labels(v6_dir)
        v7_labels = self._load_all_labels(v7_dir)

        assert set(v6_labels.keys()) == set(v7_labels.keys()), (
            f"Different files: v6={set(v6_labels.keys())}, v7={set(v7_labels.keys())}"
        )
        for name in v6_labels:
            assert v6_labels[name] == v7_labels[name], (
                f"{name} differs between v6 and v7"
            )

    def test_review_has_no_false_section_warnings(self):
        """v7 review track should not contain section-boundary artifacts
        like 'empty bars' from v6's section machinery."""
        import os
        from pathlib import Path

        v7_dir = "/Users/bwagner/projects/remap_labels/remapped_v7"
        review_path = f"{v7_dir}/review.txt"
        if not os.path.exists(review_path):
            pytest.skip("Run v7 remap_labels on real data first")

        content = Path(review_path).read_text()
        assert "empty" not in content.lower(), (
            "v7 should not have section-boundary 'empty bars' warnings"
        )


REAL_DATA_AVAILABLE = (
    os.path.exists(f"{V6_BASELINE_DIR}/chords_blues_brothers_everybody_needs_somebody.txt")
    and os.path.exists(f"{SONG_DIR}/new_bars.txt")
    and os.path.exists(f"{SONG_DIR}/new_beats.txt")
)


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="Run remap_labels on real data first")
class TestRealSongOutput:

    def test_all_chord_boundaries_on_beats(self):
        chords = _load_chords(
            f"{V6_BASELINE_DIR}/chords_blues_brothers_everybody_needs_somebody.txt"
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
            f"{V6_BASELINE_DIR}/chords_blues_brothers_everybody_needs_somebody.txt"
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
        review_path = f"{V6_BASELINE_DIR}/review.txt"
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
        review_path = f"{V6_BASELINE_DIR}/review.txt"
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
