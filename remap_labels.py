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

import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

__version__ = "0.8.0"


def get_version_info(version):
    """Return version string enriched with git commit, dirty flag, and timestamp."""
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        ts = subprocess.check_output(
            ["git", "log", "-1", "--format=%ai"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        suffix = "-dirty" if dirty else ""
        return f"{version} (git:{rev}{suffix}, {ts})"
    except Exception:
        return version


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
# Small value added to chroma features to prevent NaN in cosine distance
# from zero-magnitude frames (silence)
CHROMA_EPSILON = 1e-10


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

    chroma_old = chroma_old + CHROMA_EPSILON
    chroma_new = chroma_new + CHROMA_EPSILON

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


def parse_labels_to_bar_beat(
    labels: list[LabelEntry],
    beat_grid: list[float],
    bar_grid: list[float],
) -> list[SectionEntry]:
    """Parse labels into absolute (bar, beat) positions.

    Each label gets a bar_offset (Fraction) representing its absolute
    position in the song: bar index + beat-within-bar / beats-per-bar.
    No section logic - positions are absolute.
    """
    entries = []
    for label in labels:
        chord_bar = _find_bar_for_time(label.start, bar_grid)
        beat_in_bar = _beat_position_in_bar(
            label.start, chord_bar, beat_grid, bar_grid,
        )
        bar_beats = _beats_in_bar(chord_bar, beat_grid, bar_grid)
        beats_per_bar = max(len(bar_beats), 1)
        bar_offset = Fraction(chord_bar) + Fraction(beat_in_bar, beats_per_bar)

        if label.is_point:
            entries.append(SectionEntry(
                bar_offset=bar_offset,
                label=label.label,
                is_point=True,
                bar_count=Fraction(0),
            ))
        else:
            start_beat = grid_index(label.start, beat_grid)
            end_beat = grid_index(label.end, beat_grid)
            beat_count = max(end_beat - start_beat, 1)
            bar_count = Fraction(beat_count, beats_per_bar)
            entries.append(SectionEntry(
                bar_offset=bar_offset,
                label=label.label,
                is_point=False,
                bar_count=bar_count,
            ))
    return entries


def reconstruct_labels(
    entries: list[SectionEntry],
    beat_grid: list[float],
    bar_grid: list[float],
    beats_per_bar: int,
) -> tuple[list[LabelEntry], list[str]]:
    """Reconstruct labels on a new bar/beat grid from absolute positions.

    Each entry's bar_offset is absolute (not section-relative).
    Uses bar grid for bar positions, beat grid for sub-bar.
    """
    return reconstruct_section(
        entries, beat_grid, bar_grid, beats_per_bar,
        section_start_bar=0, section_end_bar=len(bar_grid),
    )


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


def load_labels(path: str) -> list[LabelEntry]:
    """Load an Audacity label file into LabelEntry list."""
    entries = []
    for line in Path(path).read_text().splitlines():
        parsed = parse_label_line(line)
        if parsed is not None:
            start, end, label = parsed
            entries.append(LabelEntry(start, end, label))
    return entries


def main_v7(
    old_audio: str,
    new_audio: str,
    new_beats_path: str,
    new_bars_path: str,
    old_beats_path: str,
    old_bars_path: str,
    label_files: list[str],
    outdir: str,
) -> None:
    """v7: Direct bar/beat remapping without sections.

    Each label is parsed to absolute (bar, beat), DTW finds bar 1 offset,
    labels are shifted and placed on the new grid.
    """
    # Load grids
    new_beat_grid = load_timestamps(new_beats_path)
    new_bar_grid = load_timestamps(new_bars_path)
    old_beat_grid = load_timestamps(old_beats_path)
    old_bar_grid = load_timestamps(old_bars_path)

    beats_per_bar = round(len(old_beat_grid) / max(len(old_bar_grid), 1))
    print(f"Beats per bar: {beats_per_bar}")
    print(f"New: {len(new_beat_grid)} beats, {len(new_bar_grid)} bars")
    print(f"Old: {len(old_beat_grid)} beats, {len(old_bar_grid)} bars")

    # Validate beat consistency
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

    # Compute DTW to find bar offset between old and new
    old_times, new_times = compute_alignment(old_audio, new_audio)
    warp = make_warp_func(old_times, new_times)

    # Find where old bar 0 maps to in new
    warped_first = warp(old_bar_grid[0])
    new_first_bar = grid_index(warped_first, new_bar_grid)
    bar_shift = new_first_bar - 0  # how many bars to shift
    print(f"\nBar shift: old bar 1 -> new bar {new_first_bar + 1} "
          f"(shift={bar_shift:+d})")

    # Detect structural anomalies
    print("\nChecking for structural changes...")
    anomalies = detect_anomalies(old_times, new_times)

    def time_to_bar_beat(t):
        bar_idx = _find_bar_for_time(t, new_bar_grid)
        bar_num = bar_idx + 1
        beat_pos = _beat_position_in_bar(t, bar_idx, new_beat_grid, new_bar_grid)
        if beat_pos == 0:
            return f"bar {bar_num}"
        return f"bar {bar_num} beat {beat_pos + 1}"

    if anomalies:
        print(f"\n{'='*60}")
        print("STRUCTURAL CHANGES DETECTED - manual review needed:")
        print(f"{'='*60}")
        for a in anomalies:
            if a["type"] == "added":
                print(f"  ADDED: {time_to_bar_beat(a['new_start'])} - "
                      f"{time_to_bar_beat(a['new_end'])}")
            else:
                print(f"  REMOVED: old {a['old_start']:.1f}-{a['old_end']:.1f}s")
        print(f"{'='*60}")
    else:
        print("No major structural changes detected.")

    # Seed review marks from anomalies
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

        # Parse to absolute bar/beat positions
        entries = parse_labels_to_bar_beat(
            old_labels, old_beat_grid, old_bar_grid,
        )

        # Apply bar shift
        shifted = [
            SectionEntry(
                bar_offset=e.bar_offset + bar_shift,
                label=e.label,
                is_point=e.is_point,
                bar_count=e.bar_count,
            )
            for e in entries
        ]

        # Reconstruct on new grid
        labels, warnings = reconstruct_labels(
            shifted, new_beat_grid, new_bar_grid, beats_per_bar,
        )

        # Write output
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_lines = [
            format_label(le.start, le.end, le.label) for le in labels
        ]
        Path(out_path).write_text("\n".join(out_lines) + "\n")
        print(f"  {name} -> {out_path} ({len(labels)} labels)")

        all_warnings.extend(warnings)
        for w in warnings:
            # Extract bar number from warning for review mark positioning
            all_review_marks.append((0.0, w))

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
            print(f"  {w}")

    print(f"\nDone. Check {outdir}/ and verify in Audacity.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remap Audacity labels from old audio to new audio."
    )
    parser.add_argument(
        "-V", "--version", action="version",
        version=get_version_info(__version__),
    )
    parser.add_argument("old_audio", help="Original audio file")
    parser.add_argument("new_audio", help="New (replacement) audio file")
    parser.add_argument(
        "labels", nargs="*", help="Label .txt files to remap"
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
        "-b", "--new-beats", required=True,
        help="New beats file",
    )
    parser.add_argument(
        "-B", "--new-bars", required=True,
        help="New bars file",
    )
    parser.add_argument(
        "-o", "--outdir", default="remapped",
        help="Output directory (default: remapped/)",
    )
    args = parser.parse_args()

    if not args.labels:
        print("Error: provide at least one label file to remap.", file=sys.stderr)
        sys.exit(1)

    main_v7(
        args.old_audio,
        args.new_audio,
        args.new_beats,
        args.new_bars,
        args.old_beats,
        args.old_bars,
        args.labels,
        args.outdir,
    )
