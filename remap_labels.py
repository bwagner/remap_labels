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

import librosa
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
    """Compute DTW alignment. Returns (warp_func, old_times, new_times) from the path."""
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
        s, _e, l = result[i]
        next_s = result[i + 1][0]
        result[i] = (s, next_s, l)

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
        s, _e, l = result[i]
        next_s = result[i + 1][0]
        result[i] = (s, next_s, l)

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
            for s, e, l in remapped:
                out_lines.append(format_label(s, e, l))
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
        print(f"STRUCTURAL CHANGES DETECTED - manual review needed:")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid DTW + beat-grid label remapping for Audacity."
    )
    parser.add_argument("old_audio", help="Original audio file")
    parser.add_argument("new_audio", help="New (replacement) audio file")
    parser.add_argument(
        "labels", nargs="*", help="Label .txt files to remap"
    )
    parser.add_argument(
        "-b", "--new-beats", required=True,
        help="New beats file (from DBNBeatTracker on new audio)",
    )
    parser.add_argument(
        "-B", "--new-bars",
        help="New bars file (from beats2bars.py on new beats)",
    )
    parser.add_argument(
        "--old-beats",
        help="Old beats file - enables beat-count preservation for chains",
    )
    parser.add_argument(
        "--old-bars",
        help="Old bars file - ensures chain starts stay on bar boundaries",
    )
    parser.add_argument(
        "-s", "--bar-snap", action="append", default=[],
        help="Label file(s) to snap to bars instead of beats (repeatable)",
    )
    parser.add_argument(
        "-o", "--outdir", default="remapped",
        help="Output directory (default: remapped/)",
    )
    args = parser.parse_args()

    if not args.labels:
        print("Error: provide at least one label file to remap.", file=sys.stderr)
        sys.exit(1)

    bar_snap_files = set(args.bar_snap)

    main(
        args.old_audio,
        args.new_audio,
        args.new_beats,
        args.new_bars,
        args.old_beats,
        args.old_bars,
        args.labels,
        bar_snap_files,
        args.outdir,
    )
