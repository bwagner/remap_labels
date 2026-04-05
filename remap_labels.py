#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "librosa>=0.11",
#     "soundfile",
#     "numpy",
# ]
# ///
"""Hybrid label remapping: DTW alignment + beat-count preservation.

When --old-beats is provided, chord/label durations are measured in old beats
and preserved on the new grid. DTW positions the chain start; original beat
counts lay out the rest. Re-anchors via DTW when drift exceeds a threshold,
to handle structural changes (added/removed sections).

Without --old-beats, falls back to independent DTW snap per boundary.

Usage:
    remap_labels.py old.mp3 new.mp3 \\
        -b new_beats.txt -B new_bars.txt \\
        --old-beats old_beats.txt \\
        -s parts.txt \\
        chords.txt parts.txt guit.txt

Output goes to <outdir>/<original_name> (default: remapped/).
"""

import argparse
import sys
from pathlib import Path

from dataclasses import dataclass
from fractions import Fraction

import numpy as np


SAMPLE_RATE = 22050
HOP_DURATION_S = 0.01
HOP_LENGTH = int(SAMPLE_RATE * HOP_DURATION_S)

# Thresholds for detecting structural changes (added/removed sections)
# If DTW maps N seconds of old audio to < COMPRESS_RATIO * N in new, section may be removed
COMPRESS_RATIO = 0.3
# If DTW maps N seconds of old audio to > EXPAND_RATIO * N in new, section may be added
EXPAND_RATIO = 3.0
# Minimum duration (seconds) of a structural anomaly to report
MIN_ANOMALY_DURATION_S = 2.0
# Window size (seconds) for scanning the DTW path for anomalies
SCAN_WINDOW_S = 4.0


def load_timestamps(path: str) -> list[float]:
    """Load timestamps (first column) from an Audacity label file."""
    times = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if parts:
            try:
                times.append(float(parts[0]))
            except ValueError:
                continue
    return sorted(set(times))


def compute_alignment(old_audio: str, new_audio: str):
    """Compute DTW alignment. Returns (old_times, new_times) from the path."""
    import librosa

    print(f"Loading old audio: {old_audio}")
    y_old, _ = librosa.load(old_audio, sr=SAMPLE_RATE)
    print(f"Loading new audio: {new_audio}")
    y_new, _ = librosa.load(new_audio, sr=SAMPLE_RATE)

    print("Computing chroma features...")
    chroma_old = librosa.feature.chroma_cqt(y=y_old, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    chroma_new = librosa.feature.chroma_cqt(y=y_new, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

    print(f"Running DTW ({chroma_old.shape[1]} x {chroma_new.shape[1]} frames)...")
    _D, wp = librosa.sequence.dtw(X=chroma_old, Y=chroma_new, metric="cosine")
    wp = wp[::-1]  # sort ascending

    old_frames = wp[:, 0]
    new_frames = wp[:, 1]
    old_times = librosa.frames_to_time(old_frames, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    new_times = librosa.frames_to_time(new_frames, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

    print(f"Alignment: {old_times[-1]:.1f}s old -> {new_times[-1]:.1f}s new")
    return old_times, new_times


def make_warp_func(old_times: np.ndarray, new_times: np.ndarray):
    """Return a function mapping old timestamp -> new timestamp via interpolation."""

    def warp(t: float) -> float:
        if t <= old_times[0]:
            return float(new_times[0])
        if t >= old_times[-1]:
            return float(new_times[-1])
        return float(np.interp(t, old_times, new_times))

    return warp


def snap_to_grid(t: float, grid: list[float]) -> float:
    """Snap timestamp to nearest point in grid."""
    idx = np.searchsorted(grid, t)
    candidates = []
    if idx > 0:
        candidates.append(grid[idx - 1])
    if idx < len(grid):
        candidates.append(grid[idx])
    if not candidates:
        return t
    return min(candidates, key=lambda g: abs(g - t))


def detect_anomalies(
    old_times: np.ndarray, new_times: np.ndarray
) -> list[dict]:
    """Detect likely added/removed sections by scanning the DTW path.

    Returns list of {type: 'added'|'removed', old_start, old_end, new_start, new_end}.
    """
    anomalies = []
    # Sample the path at regular intervals
    step = SCAN_WINDOW_S
    max_old = old_times[-1]
    t = 0.0
    warp = make_warp_func(old_times, new_times)
    inv_warp = make_warp_func(new_times, old_times)

    # Scan for compressions (removed from new)
    while t < max_old - step:
        old_start, old_end = t, t + step
        new_start, new_end = warp(old_start), warp(old_end)
        old_span = old_end - old_start
        new_span = new_end - new_start
        if old_span > MIN_ANOMALY_DURATION_S and new_span < COMPRESS_RATIO * old_span:
            anomalies.append({
                "type": "removed",
                "old_start": old_start,
                "old_end": old_end,
                "new_start": new_start,
                "new_end": new_end,
            })
            t = old_end  # skip past this anomaly
        else:
            t += step / 2

    # Scan for expansions (added in new)
    max_new = new_times[-1]
    t = 0.0
    while t < max_new - step:
        new_start, new_end = t, t + step
        old_start, old_end = inv_warp(new_start), inv_warp(new_end)
        old_span = old_end - old_start
        new_span = new_end - new_start
        if new_span > MIN_ANOMALY_DURATION_S and old_span < COMPRESS_RATIO * new_span:
            anomalies.append({
                "type": "added",
                "old_start": old_start,
                "old_end": old_end,
                "new_start": new_start,
                "new_end": new_end,
            })
            t = new_end
        else:
            t += step / 2

    return anomalies


def parse_label_line(line: str) -> tuple[float, float, str] | None:
    """Parse one Audacity label line: start\\tend\\tlabel"""
    line = line.strip()
    if not line:
        return None
    parts = line.split("\t")
    if len(parts) < 2:
        return None
    start = float(parts[0])
    end = float(parts[1])
    label = parts[2] if len(parts) > 2 else ""
    return start, end, label


def format_label(start: float, end: float, label: str) -> str:
    return f"{start:.6f}\t{end:.6f}\t{label}"


def grid_index(t: float, grid: list[float]) -> int:
    """Return index of nearest grid point."""
    idx = int(np.searchsorted(grid, t))
    if idx == 0:
        return 0
    if idx >= len(grid):
        return len(grid) - 1
    if abs(grid[idx] - t) < abs(grid[idx - 1] - t):
        return idx
    return idx - 1


# Tolerance for detecting chained labels (end of one == start of next)
CHAIN_TOLERANCE = 0.02

# Tolerance for matching a timestamp to a grid point
GRID_MATCH_TOLERANCE = 0.05


# -- v6: Musical-structure data types and functions --


@dataclass
class LabelEntry:
    """A parsed Audacity label."""

    start: float
    end: float
    label: str

    @property
    def is_point(self) -> bool:
        return abs(self.end - self.start) < 0.001


@dataclass
class SectionEntry:
    """A label entry relative to its section, measured in bars (Fraction)."""

    bar_offset: Fraction
    label: str
    is_point: bool
    bar_count: Fraction  # Fraction(0) for point labels


@dataclass
class MappedSection:
    """A section mapped to the new bar grid."""

    name: str
    new_bar_idx: int  # index into new_bar_grid


def _beats_in_bar(bar_idx: int, beat_grid: list[float], bar_grid: list[float]) -> list[int]:
    """Return beat grid indices that fall within the given bar."""
    bar_start = bar_grid[bar_idx]
    bar_end = bar_grid[bar_idx + 1] if bar_idx + 1 < len(bar_grid) else float("inf")
    result = []
    start_beat = grid_index(bar_start, beat_grid)
    for i in range(start_beat, len(beat_grid)):
        if beat_grid[i] >= bar_end - GRID_MATCH_TOLERANCE:
            break
        if beat_grid[i] >= bar_start - GRID_MATCH_TOLERANCE:
            result.append(i)
    return result


def _find_bar_for_time(t: float, bar_grid: list[float]) -> int:
    """Return the 0-based index of the bar that contains time t."""
    idx = grid_index(t, bar_grid)
    if idx < len(bar_grid) - 1 and bar_grid[idx] > t + GRID_MATCH_TOLERANCE:
        idx = max(idx - 1, 0)
    return idx


def _beat_position_in_bar(
    t: float, bar_idx: int, beat_grid: list[float], bar_grid: list[float],
) -> int:
    """Return which beat (0-indexed) within a bar the time t falls on."""
    bar_beats = _beats_in_bar(bar_idx, beat_grid, bar_grid)
    for pos, beat_idx in enumerate(bar_beats):
        if abs(beat_grid[beat_idx] - t) < GRID_MATCH_TOLERANCE:
            return pos
    return 0


def parse_section_entries(
    chords: list[LabelEntry],
    beat_grid: list[float],
    bar_grid: list[float],
    beats_per_bar: int,
    section_start: float,
    section_end: float,
) -> list[SectionEntry]:
    """Extract chord/label entries within a section as bar offsets.

    Uses the bar grid directly: bar_offset = (chord's bar index) - (section start bar index).
    Sub-bar position from beat position within the bar.
    Duration from beat counting, expressed in bars.
    """
    section_start_bar = _find_bar_for_time(section_start, bar_grid)
    entries = []
    for chord in chords:
        if chord.start < section_start - GRID_MATCH_TOLERANCE:
            continue
        if chord.start >= section_end - GRID_MATCH_TOLERANCE:
            continue

        chord_bar = _find_bar_for_time(chord.start, bar_grid)
        rel_bar = chord_bar - section_start_bar
        beat_in_bar = _beat_position_in_bar(
            chord.start, chord_bar, beat_grid, bar_grid,
        )
        bar_offset = Fraction(rel_bar) + Fraction(beat_in_bar, beats_per_bar)

        if chord.is_point:
            entries.append(SectionEntry(
                bar_offset=bar_offset,
                label=chord.label,
                is_point=True,
                bar_count=Fraction(0),
            ))
        else:
            start_beat = grid_index(chord.start, beat_grid)
            end_beat = grid_index(chord.end, beat_grid)
            beat_count = max(end_beat - start_beat, 1)
            bar_count = Fraction(beat_count, beats_per_bar)
            entries.append(SectionEntry(
                bar_offset=bar_offset,
                label=chord.label,
                is_point=False,
                bar_count=bar_count,
            ))
    return entries


def reconstruct_section(
    entries: list[SectionEntry],
    beat_grid: list[float],
    bar_grid: list[float],
    beats_per_bar: int,
    section_start_bar: int,
    section_end_bar: int,
) -> tuple[list[LabelEntry], list[str]]:
    """Reconstruct label entries on new bar/beat grids.

    Positions chords using bar grid (whole bars) + beat grid (sub-bar).
    A chord at bar_offset 21.5 goes at: bar_grid[start + 21], beat 2.

    Returns (labels, warnings).
    """
    warnings = []
    labels = []
    for entry in entries:
        whole_bars = int(entry.bar_offset)
        frac_beats = int((entry.bar_offset - whole_bars) * beats_per_bar)
        abs_bar = section_start_bar + whole_bars

        if abs_bar >= len(bar_grid) or abs_bar >= section_end_bar:
            warnings.append(
                f"'{entry.label}' at bar {entry.bar_offset} dropped"
            )
            continue

        # Find beat within bar
        bar_beats = _beats_in_bar(abs_bar, beat_grid, bar_grid)
        if frac_beats < len(bar_beats):
            start_time = beat_grid[bar_beats[frac_beats]]
        else:
            start_time = bar_grid[abs_bar]

        if entry.is_point:
            labels.append(LabelEntry(start_time, start_time, entry.label))
            continue

        # Compute end time using bar_count
        end_whole_bars = int(entry.bar_count)
        end_frac_beats = int((entry.bar_count - end_whole_bars) * beats_per_bar)
        end_abs_bar = abs_bar + end_whole_bars

        # Determine end beat position
        end_beat_offset = frac_beats + end_frac_beats
        end_bar_from_beat = end_beat_offset // beats_per_bar
        end_beat_in_bar = end_beat_offset % beats_per_bar
        end_abs_bar += end_bar_from_beat

        if end_abs_bar > section_end_bar:
            if abs_bar < section_end_bar:
                end_time = bar_grid[section_end_bar] if section_end_bar < len(bar_grid) else beat_grid[-1]
                warnings.append(
                    f"'{entry.label}' at bar {entry.bar_offset} "
                    f"(+{entry.bar_count} bars) truncated to section end"
                )
            else:
                warnings.append(
                    f"'{entry.label}' at bar {entry.bar_offset} "
                    f"starts beyond section end (dropped)"
                )
                continue
        elif end_abs_bar >= len(bar_grid):
            warnings.append(
                f"'{entry.label}' at bar {entry.bar_offset} "
                f"exceeds grid (dropped)"
            )
            continue
        elif end_beat_in_bar == 0:
            # Ends on a bar boundary
            end_time = bar_grid[end_abs_bar]
        else:
            end_bar_beats = _beats_in_bar(end_abs_bar, beat_grid, bar_grid)
            if end_beat_in_bar < len(end_bar_beats):
                end_time = beat_grid[end_bar_beats[end_beat_in_bar]]
            else:
                end_time = bar_grid[end_abs_bar]

        labels.append(LabelEntry(start_time, end_time, entry.label))

    # Detect empty bars at end of section
    range_labels = [lbl for lbl in labels if not lbl.is_point]
    if range_labels and section_end_bar < len(bar_grid):
        last_end = max(lbl.end for lbl in range_labels)
        last_bar = _find_bar_for_time(last_end, bar_grid)
        if last_bar < section_end_bar - 1:
            empty_start = last_bar + 2  # 1-indexed
            empty_end = section_end_bar  # 1-indexed
            warnings.append(
                f"bars {empty_start}-{empty_end} empty "
                f"(section has {section_end_bar - last_bar - 1} extra bars)"
            )

    return labels, warnings


def validate_bar_beats(
    old_beat_grid: list[float],
    old_bar_grid: list[float],
    new_beat_grid: list[float],
    new_bar_grid: list[float],
) -> list[str]:
    """Compare beats-per-bar between old and new grids.

    Returns warnings for bar count mismatches and per-bar beat count
    differences.
    """
    warnings = []

    if len(old_bar_grid) != len(new_bar_grid):
        warnings.append(
            f"Bar count mismatch: old has {len(old_bar_grid)} bars, "
            f"new has {len(new_bar_grid)} bars"
        )

    n_bars = min(len(old_bar_grid), len(new_bar_grid))
    for i in range(n_bars):
        old_beats = _beats_in_bar(i, old_beat_grid, old_bar_grid)
        new_beats = _beats_in_bar(i, new_beat_grid, new_bar_grid)
        if len(old_beats) != len(new_beats):
            warnings.append(
                f"Bar {i + 1}: old has {len(old_beats)} beats, "
                f"new has {len(new_beats)} beats"
            )

    return warnings


def _old_section_bar_count(
    section_start: float,
    section_end: float,
    bar_grid: list[float],
) -> int:
    """Count how many bars an old section spans."""
    start_bar = _find_bar_for_time(section_start, bar_grid)
    end_bar = _find_bar_for_time(section_end, bar_grid)
    return max(end_bar - start_bar, 1)


def map_section_boundaries(
    old_parts: list[LabelEntry],
    warp: callable,
    new_bar_grid: list[float],
    old_bar_grid: list[float] | None = None,
) -> list[MappedSection]:
    """Map old section starts to new bar positions.

    Uses DTW to position the first section, then places subsequent
    sections contiguously based on old section bar counts. This
    prevents gaps between sections caused by DTW imprecision.
    """
    if not old_parts:
        return []

    # DTW for the first section
    warped = warp(old_parts[0].start)
    first_bar = grid_index(warped, new_bar_grid)
    result = [MappedSection(name=old_parts[0].label, new_bar_idx=first_bar)]

    # Subsequent sections: place contiguously
    cursor = first_bar
    for i in range(1, len(old_parts)):
        if old_bar_grid is not None:
            prev_bars = _old_section_bar_count(
                old_parts[i - 1].start, old_parts[i - 1].end, old_bar_grid,
            )
        else:
            # Fallback to DTW if no old bar grid
            warped = warp(old_parts[i].start)
            prev_bars = grid_index(warped, new_bar_grid) - cursor
        cursor += prev_bars
        cursor = min(cursor, len(new_bar_grid) - 1)
        result.append(MappedSection(name=old_parts[i].label, new_bar_idx=cursor))

    return result

# Tolerance for detecting if a timestamp falls on a bar boundary
BAR_ALIGN_TOLERANCE = 0.05


def find_chain_segments(
    labels: list[tuple[int, float, float, str]],
) -> list[list[tuple[int, float, float, str]]]:
    """Split labels into chain segments and isolated labels.

    Input: list of (original_index, start, end, label).
    A chain segment is a maximal run of consecutive range labels where
    each end matches the next start within tolerance.
    Returns a list of segments (each segment is a list of labels).
    """
    segments = []
    current_chain = []

    for item in labels:
        idx, start, end, label = item
        is_point = abs(end - start) < 0.001

        if is_point:
            # Flush current chain
            if current_chain:
                segments.append(current_chain)
                current_chain = []
            segments.append([item])  # isolated point label
        else:
            if current_chain:
                prev_end = current_chain[-1][2]  # end of previous
                if abs(prev_end - start) <= CHAIN_TOLERANCE:
                    current_chain.append(item)
                else:
                    # Gap - flush chain, start new
                    segments.append(current_chain)
                    current_chain = [item]
            else:
                current_chain = [item]

    if current_chain:
        segments.append(current_chain)

    return segments


# If DTW position drifts more than this many beats from the beat-count
# layout, re-anchor to the DTW position (likely a structural change).
REANCHOR_THRESHOLD_BEATS = 4


def compute_beat_counts(
    labels: list[tuple[int, float, float, str]],
    old_beat_grid: list[float],
) -> list[int]:
    """Compute each label's duration in old beats (rounded to nearest int, min 1)."""
    counts = []
    for _idx, start, end, _label in labels:
        si = grid_index(start, old_beat_grid)
        ei = grid_index(end, old_beat_grid)
        counts.append(max(ei - si, 1))
    return counts


def remap_chain_segment(
    labels: list[tuple[int, float, float, str]],
    warp: callable,
    grid: list[float],
    old_beat_grid: list[float] | None = None,
    old_bar_grid: list[float] | None = None,
    new_bar_grid: list[float] | None = None,
) -> tuple[list[tuple[float, float, str]], list[str]]:
    """Remap a contiguous chain of range labels, snapping to grid.

    If old_beat_grid is provided, preserves original beat counts per label
    and only uses DTW to position the chain start (plus periodic re-anchoring).
    When old/new bar grids are available, chain starts that were on bar
    boundaries are snapped to new bar boundaries (prevents +-1 beat shift).
    Otherwise falls back to independent DTW snapping per boundary.

    Returns (remapped_labels, warnings).
    """
    warnings = []
    n = len(labels)

    if old_beat_grid is not None:
        return _remap_chain_preserving_beats(
            labels, warp, grid, old_beat_grid, old_bar_grid, new_bar_grid
        )

    # Fallback: independent DTW snap per boundary
    warped_indices = []
    for orig_idx, start, end, label in labels:
        warped_start = warp(start)
        gi = grid_index(warped_start, grid)
        warped_indices.append((gi, label, start, end))

    last_end = labels[-1][2]
    warped_last_end = warp(last_end)
    last_end_gi = grid_index(warped_last_end, grid)

    needed = n
    first_gi = warped_indices[0][0]
    available = max(last_end_gi - first_gi, 0)
    if available < needed:
        last_end_gi = min(first_gi + needed, len(grid) - 1)

    result = []
    cursor = warped_indices[0][0]

    for i, (gi, label, old_start, old_end) in enumerate(warped_indices):
        snap_start_idx = max(gi, cursor)
        if snap_start_idx >= len(grid) - 1:
            warnings.append(
                f"  WARNING: '{label}' ({old_start:.2f}-{old_end:.2f}s) "
                f"ran off end of grid, skipping"
            )
            continue

        if i + 1 < n:
            next_gi = warped_indices[i + 1][0]
            snap_end_idx = max(next_gi, snap_start_idx + 1)
        else:
            snap_end_idx = max(last_end_gi, snap_start_idx + 1)

        snap_end_idx = min(snap_end_idx, len(grid) - 1)
        if snap_start_idx >= snap_end_idx:
            snap_end_idx = min(snap_start_idx + 1, len(grid) - 1)

        result.append((grid[snap_start_idx], grid[snap_end_idx], label))
        cursor = snap_end_idx

    for i in range(len(result) - 1):
        start, _end, lbl = result[i]
        next_start = result[i + 1][0]
        result[i] = (start, next_start, lbl)

    return result, warnings, []  # no review marks for fallback path


def _is_on_bar(t: float, bar_grid: list[float] | None) -> bool:
    """Check if timestamp t falls on a bar boundary."""
    if not bar_grid:
        return False
    nearest = snap_to_grid(t, bar_grid)
    return abs(nearest - t) < BAR_ALIGN_TOLERANCE


def _snap_to_bar_on_grid(
    warped_t: float, grid: list[float], bar_grid: list[float],
) -> int:
    """Snap warped time to nearest bar boundary, return its index in grid."""
    nearest_bar = snap_to_grid(warped_t, bar_grid)
    return grid_index(nearest_bar, grid)


def _remap_chain_preserving_beats(
    labels: list[tuple[int, float, float, str]],
    warp: callable,
    grid: list[float],
    old_beat_grid: list[float],
    old_bar_grid: list[float] | None = None,
    new_bar_grid: list[float] | None = None,
) -> tuple[list[tuple[float, float, str]], list[str]]:
    """Remap chain using original beat counts.

    1. Compute each label's duration in old beats.
    2. DTW-warp the chain start to the new grid.
       If the original start was on a bar boundary, snap to new bar boundary.
    3. Lay out labels using original beat counts.
    4. Re-anchor via DTW when accumulated drift exceeds threshold.
       Re-anchors also snap to bar boundaries when appropriate.
    """
    warnings = []
    beat_counts = compute_beat_counts(labels, old_beat_grid)

    # Position the first label's start via DTW
    first_start = labels[0][1]
    warped_first = warp(first_start)

    # If original chain started on a bar boundary, snap to new bar boundary
    if _is_on_bar(first_start, old_bar_grid) and new_bar_grid:
        cursor = _snap_to_bar_on_grid(warped_first, grid, new_bar_grid)
    else:
        cursor = grid_index(warped_first, grid)

    result = []
    review_marks = []  # (time, label) for the review track

    for i, (_idx, start, end, label) in enumerate(labels):
        snap_start_idx = cursor

        if snap_start_idx >= len(grid) - 1:
            warnings.append(
                f"  WARNING: '{label}' ({start:.2f}-{end:.2f}s) "
                f"ran off end of grid, skipping"
            )
            continue

        snap_end_idx = snap_start_idx + beat_counts[i]
        if snap_end_idx >= len(grid):
            snap_end_idx = len(grid) - 1
            if snap_start_idx >= snap_end_idx:
                warnings.append(
                    f"  WARNING: '{label}' ({start:.2f}-{end:.2f}s) "
                    f"ran off end of grid, skipping"
                )
                continue

        result.append((grid[snap_start_idx], grid[snap_end_idx], label))
        cursor = snap_end_idx

        # Gradual drift correction: nudge by at most 1 beat per label
        if i + 1 < len(labels):
            next_start = labels[i + 1][1]
            dtw_next = warp(next_start)
            dtw_gi = grid_index(dtw_next, grid)
            # Bar-align the DTW target if original was on a bar
            if _is_on_bar(next_start, old_bar_grid) and new_bar_grid:
                dtw_gi = _snap_to_bar_on_grid(dtw_next, grid, new_bar_grid)
            drift = dtw_gi - cursor  # positive = cursor is behind DTW
            if drift > 0:
                # Cursor behind DTW: stretch next label by 1 beat
                cursor += 1
                review_marks.append((
                    grid[cursor],
                    f"+1 beat after '{label}' (drift={drift})",
                ))
            elif drift < -1:
                # Cursor ahead of DTW: shrink next label by 1 beat
                cursor -= 1
                review_marks.append((
                    grid[cursor],
                    f"-1 beat after '{label}' (drift={drift})",
                ))

    # Enforce exact chain
    for i in range(len(result) - 1):
        start, _end, lbl = result[i]
        next_start = result[i + 1][0]
        result[i] = (start, next_start, lbl)

    return result, warnings, review_marks


def remap_label_file(
    label_path: str,
    warp: callable,
    beat_grid: list[float],
    bar_grid: list[float] | None,
    old_beat_grid: list[float] | None,
    old_bar_grid: list[float] | None,
    use_bar_snap: bool,
    out_path: str,
) -> tuple[list[str], list[tuple[float, str]]]:
    """Remap a label file. Returns (warnings, review_marks)."""
    grid = bar_grid if (use_bar_snap and bar_grid) else beat_grid
    # Beat-count preservation only makes sense when snapping to beats.
    # At bar resolution, DTW-only snapping is accurate enough.
    old_grid_for_counts = None if use_bar_snap else old_beat_grid
    warnings = []
    review_marks = []
    lines = Path(label_path).read_text().splitlines()

    # Parse all labels with their original indices
    indexed_labels = []
    for i, line in enumerate(lines):
        parsed = parse_label_line(line)
        if parsed is not None:
            start, end, label = parsed
            indexed_labels.append((i, start, end, label))

    if not indexed_labels:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("\n".join(lines) + "\n")
        return warnings, review_marks

    # Split into chain segments and isolated labels
    segments = find_chain_segments(indexed_labels)
    out_lines = []

    for seg in segments:
        first_item = seg[0]
        is_point = abs(first_item[2] - first_item[1]) < 0.001
        is_chain = len(seg) > 1 and not is_point

        if is_chain:
            remapped, seg_warnings, seg_reviews = remap_chain_segment(
                seg, warp, grid, old_grid_for_counts,
                old_bar_grid, bar_grid,
            )
            warnings.extend(seg_warnings)
            review_marks.extend(seg_reviews)
            for start, end, lbl in remapped:
                out_lines.append(format_label(start, end, lbl))
        else:
            # Single label (point or isolated range)
            for _idx, start, end, label in seg:
                warped_start = warp(start)
                snapped_start = snap_to_grid(warped_start, grid)

                if is_point:
                    out_lines.append(format_label(snapped_start, snapped_start, label))
                else:
                    warped_end = warp(end)
                    snapped_end = snap_to_grid(warped_end, grid)
                    if snapped_end <= snapped_start:
                        gi = grid_index(snapped_start, grid)
                        if gi + 1 < len(grid):
                            snapped_end = grid[gi + 1]
                        else:
                            snapped_end = snapped_start + 0.01
                    out_lines.append(format_label(snapped_start, snapped_end, label))

                drift = abs(warped_start - snapped_start)
                if drift > 1.0:
                    warnings.append(
                        f"  WARNING: '{label}' snapped {drift:.2f}s from DTW "
                        f"({warped_start:.2f} -> {snapped_start:.2f})"
                    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(out_lines) + "\n")
    return warnings, review_marks


def main(
    old_audio: str,
    new_audio: str,
    new_beats: str,
    new_bars: str | None,
    old_beats: str | None,
    old_bars: str | None,
    label_files: list[str],
    bar_snap_files: set[str],
    outdir: str,
) -> None:
    # Load new timing grids
    beat_grid = load_timestamps(new_beats)
    bar_grid = load_timestamps(new_bars) if new_bars else None
    old_beat_grid = load_timestamps(old_beats) if old_beats else None
    old_bar_grid = load_timestamps(old_bars) if old_bars else None
    print(f"New beat grid: {len(beat_grid)} beats")
    if bar_grid:
        print(f"New bar grid: {len(bar_grid)} bars")
    if old_beat_grid:
        print(f"Old beat grid: {len(old_beat_grid)} beats (beat-count preservation ON)")
    else:
        print("No old beats provided - using DTW-only snapping")
    if old_bar_grid:
        print(f"Old bar grid: {len(old_bar_grid)} bars (bar-boundary alignment ON)")

    # Compute DTW alignment
    old_times, new_times = compute_alignment(old_audio, new_audio)
    warp = make_warp_func(old_times, new_times)

    # Detect structural anomalies
    print("\nChecking for structural changes (added/removed sections)...")
    anomalies = detect_anomalies(old_times, new_times)
    if anomalies:
        print(f"\n{'='*60}")
        print("STRUCTURAL CHANGES DETECTED - manual review needed:")
        print(f"{'='*60}")
        for a in anomalies:
            if a["type"] == "added":
                print(
                    f"  ADDED section: new audio {a['new_start']:.1f}-{a['new_end']:.1f}s "
                    f"(no match in old audio around {a['old_start']:.1f}s)"
                )
            else:
                print(
                    f"  REMOVED section: old audio {a['old_start']:.1f}-{a['old_end']:.1f}s "
                    f"(compressed to {a['new_start']:.1f}-{a['new_end']:.1f}s in new)"
                )
        print(f"{'='*60}\n")
    else:
        print("No major structural changes detected.\n")

    # Remap each label file
    print(f"Remapping {len(label_files)} label file(s):")
    all_warnings = []
    all_review_marks = []
    for lf in label_files:
        name = Path(lf).name
        use_bar = name in bar_snap_files or lf in bar_snap_files
        out_path = str(Path(outdir) / name)
        grid_type = "bars" if use_bar else "beats"
        warnings, review_marks = remap_label_file(
            lf, warp, beat_grid, bar_grid, old_beat_grid, old_bar_grid,
            use_bar, out_path,
        )
        print(f"  {name} -> {out_path} (snapped to {grid_type})")
        all_warnings.extend(warnings)
        all_review_marks.extend(review_marks)

    # Write review label track if there are any correction points
    if all_review_marks:
        review_path = str(Path(outdir) / "review.txt")
        review_lines = []
        for t, desc in sorted(all_review_marks):
            review_lines.append(f"{t:.6f}\t{t:.6f}\t{desc}")
        Path(review_path).write_text("\n".join(review_lines) + "\n")
        print(f"\n  review track -> {review_path} ({len(all_review_marks)} marks)")

    if all_warnings:
        print(f"\n{'='*60}")
        print("WARNINGS:")
        print(f"{'='*60}")
        for w in all_warnings:
            print(w)

    print(f"\nDone. Check {outdir}/ and verify in Audacity.")


def load_labels(path: str) -> list[LabelEntry]:
    """Load an Audacity label file into LabelEntry list."""
    entries = []
    for line in Path(path).read_text().splitlines():
        parsed = parse_label_line(line)
        if parsed is not None:
            start, end, label = parsed
            entries.append(LabelEntry(start, end, label))
    return entries


def main_v6(
    old_audio: str,
    new_audio: str,
    new_beats_path: str,
    new_bars_path: str,
    old_beats_path: str,
    old_bars_path: str,
    old_parts_path: str,
    label_files: list[str],
    outdir: str,
) -> None:
    """v6: Musical-structure reconstruction.

    Uses DTW only at section level. Within sections, reconstructs
    chord/label patterns from bar offsets on the new grid.
    """
    # Load grids
    new_beat_grid = load_timestamps(new_beats_path)
    new_bar_grid = load_timestamps(new_bars_path)
    old_beat_grid = load_timestamps(old_beats_path)
    old_bar_grid = load_timestamps(old_bars_path)
    old_parts = load_labels(old_parts_path)

    beats_per_bar = round(len(old_beat_grid) / max(len(old_bar_grid), 1))
    print(f"Beats per bar: {beats_per_bar}")
    print(f"New: {len(new_beat_grid)} beats, {len(new_bar_grid)} bars")
    print(f"Old: {len(old_beat_grid)} beats, {len(old_bar_grid)} bars")
    print(f"Old parts: {len(old_parts)} sections")

    # Validate beat consistency between old and new
    val_warnings = validate_bar_beats(
        old_beat_grid, old_bar_grid, new_beat_grid, new_bar_grid,
    )
    if val_warnings:
        print(f"\n{'='*60}")
        print("GRID VALIDATION WARNINGS:")
        print(f"{'='*60}")
        for w in val_warnings:
            print(f"  {w}")
        print(f"{'='*60}")

    # Compute DTW alignment
    old_times, new_times = compute_alignment(old_audio, new_audio)
    warp = make_warp_func(old_times, new_times)

    # Map section boundaries to new bars
    mapped_sections = map_section_boundaries(old_parts, warp, new_bar_grid, old_bar_grid)
    print("\nSection mapping:")
    for ms, op in zip(mapped_sections, old_parts):
        print(f"  {ms.name}: old {op.start:.1f}s -> new bar {ms.new_bar_idx + 1} ({new_bar_grid[ms.new_bar_idx]:.1f}s)")

    # Detect structural anomalies
    print("\nChecking for structural changes...")
    anomalies = detect_anomalies(old_times, new_times)
    if anomalies:
        print(f"\n{'='*60}")
        print("STRUCTURAL CHANGES DETECTED - manual review needed:")
        print(f"{'='*60}")
        for a in anomalies:
            if a["type"] == "added":
                print(
                    f"  ADDED: new audio {a['new_start']:.1f}-{a['new_end']:.1f}s"
                )
            else:
                print(
                    f"  REMOVED: old audio {a['old_start']:.1f}-{a['old_end']:.1f}s"
                )
        print(f"{'='*60}")
    else:
        print("No major structural changes detected.")

    def time_to_bar_beat(t):
        """Convert a time to 'bar N beat K' string using new grids."""
        bar_idx = _find_bar_for_time(t, new_bar_grid)
        bar_num = bar_idx + 1
        beat_pos = _beat_position_in_bar(t, bar_idx, new_beat_grid, new_bar_grid)
        if beat_pos == 0:
            return f"bar {bar_num}"
        return f"bar {bar_num} beat {beat_pos + 1}"

    # Seed review marks from structural anomalies
    all_review_marks = []
    for a in anomalies:
        start_bb = time_to_bar_beat(a["new_start"])
        if a["type"] == "added":
            end_bb = time_to_bar_beat(a["new_end"])
            all_review_marks.append((
                a["new_start"],
                f"new has extra section {start_bb} - {end_bb}",
            ))
        else:
            end_bb = time_to_bar_beat(a["new_end"])
            all_review_marks.append((
                a["new_start"],
                f"old section missing from new, was here ({start_bb} - {end_bb})",
            ))

    # Process each label file
    print(f"\nReconstructing {len(label_files)} label file(s):")
    all_warnings = []

    for lf in label_files:
        name = Path(lf).name
        out_path = str(Path(outdir) / name)
        old_labels = load_labels(lf)

        out_labels = []
        file_warnings = []

        for sec_idx, (ms, op) in enumerate(zip(mapped_sections, old_parts)):
            # Parse entries in this section from old labels
            entries = parse_section_entries(
                old_labels, old_beat_grid, old_bar_grid,
                beats_per_bar, op.start, op.end,
            )
            if not entries:
                continue

            # Section end bar = next section's start bar, or end of grid
            if sec_idx + 1 < len(mapped_sections):
                sec_end_bar = mapped_sections[sec_idx + 1].new_bar_idx
            else:
                sec_end_bar = len(new_bar_grid) - 1

            # Reconstruct
            labels, warnings = reconstruct_section(
                entries, new_beat_grid, new_bar_grid, beats_per_bar,
                ms.new_bar_idx, sec_end_bar,
            )
            out_labels.extend(labels)

            # Prefix warnings with section name
            for w in warnings:
                file_warnings.append(f"  [{ms.name}] {w}")

            # Add review marks for any warnings
            for w in warnings:
                all_review_marks.append((
                    new_bar_grid[ms.new_bar_idx],
                    f"[{ms.name}] {w}",
                ))

        # Write output
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_lines = [
            format_label(le.start, le.end, le.label) for le in out_labels
        ]
        Path(out_path).write_text("\n".join(out_lines) + "\n")
        print(f"  {name} -> {out_path} ({len(out_labels)} labels)")
        all_warnings.extend(file_warnings)

    # Write review track (deduplicated)
    unique_marks = list(dict.fromkeys(all_review_marks))
    if unique_marks:
        review_path = str(Path(outdir) / "review.txt")
        review_lines = []
        for t, desc in sorted(unique_marks):
            review_lines.append(f"{t:.6f}\t{t:.6f}\t{desc}")
        Path(review_path).write_text("\n".join(review_lines) + "\n")
        print(f"\n  review track -> {review_path} ({len(unique_marks)} marks)")

    if all_warnings:
        print(f"\n{'='*60}")
        print("WARNINGS:")
        print(f"{'='*60}")
        for w in all_warnings:
            print(w)

    print(f"\nDone. Check {outdir}/ and verify in Audacity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remap Audacity labels from old audio to new audio."
    )
    parser.add_argument("old_audio", help="Original audio file")
    parser.add_argument("new_audio", help="New (replacement) audio file")
    parser.add_argument(
        "labels", nargs="*", help="Label .txt files to remap"
    )
    parser.add_argument(
        "-b", "--new-beats", required=True,
        help="New beats file",
    )
    parser.add_argument(
        "-B", "--new-bars", required=True,
        help="New bars file",
    )
    parser.add_argument(
        "--old-beats", required=True,
        help="Old beats file",
    )
    parser.add_argument(
        "--old-bars", required=True,
        help="Old bars file",
    )
    parser.add_argument(
        "-p", "--old-parts", required=True,
        help="Old parts file (section boundaries)",
    )
    parser.add_argument(
        "-o", "--outdir", default="remapped",
        help="Output directory (default: remapped/)",
    )
    args = parser.parse_args()

    if not args.labels:
        print("Error: provide at least one label file to remap.", file=sys.stderr)
        sys.exit(1)

    main_v6(
        args.old_audio,
        args.new_audio,
        args.new_beats,
        args.new_bars,
        args.old_beats,
        args.old_bars,
        args.old_parts,
        args.labels,
        args.outdir,
    )
