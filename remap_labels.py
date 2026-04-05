#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "librosa>=0.11",
#     "soundfile",
#     "numpy",
# ]
# ///
"""Remap Audacity label files from an old audio version to a new one using DTW."""

import argparse
import sys
from pathlib import Path
from typing import Callable

import librosa
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

SAMPLE_RATE = 22050
HOP_DURATION_S = 0.01
COMPRESS_RATIO = 0.3
EXPAND_RATIO = 3.0
MIN_ANOMALY_DURATION_S = 2.0
SCAN_WINDOW_S = 4.0
DRIFT_WARN_THRESHOLD_S = 1.0
CHAIN_TOLERANCE = 0.02
POINT_LABEL_TOLERANCE = 0.001
REANCHOR_THRESHOLD_BEATS = 4
BAR_ALIGN_TOLERANCE = 0.05


# ── Core logic ─────────────────────────────────────────────────────────────────


def compute_chroma(audio_path: str) -> np.ndarray:
    """Load audio and compute chroma_cqt features at HOP_DURATION_S hop."""
    hop_length = int(SAMPLE_RATE * HOP_DURATION_S)
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SAMPLE_RATE, hop_length=hop_length)
    return chroma


def compute_dtw_path(chroma_old: np.ndarray, chroma_new: np.ndarray) -> np.ndarray:
    """Run DTW and return the warping path as (N, 2) array of frame indices."""
    _cost, wp = librosa.sequence.dtw(
        chroma_old, chroma_new, metric="cosine", backtrack=True
    )
    # wp comes in reverse order (end to start); flip to chronological
    wp = wp[::-1]
    return wp


def build_time_mapping(wp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert warping path frame indices to time arrays."""
    old_times = wp[:, 0].astype(float) * HOP_DURATION_S
    new_times = wp[:, 1].astype(float) * HOP_DURATION_S
    return old_times, new_times


def warp_timestamp(t: float, old_times: np.ndarray, new_times: np.ndarray) -> float:
    """Interpolate a single old timestamp to its new-audio equivalent."""
    return float(np.interp(t, old_times, new_times))


def detect_anomalies(
    wp: np.ndarray,
) -> list[dict]:
    """Scan DTW path for structural anomalies (compressions/expansions).

    Returns a list of dicts with keys: type, old_start_s, old_end_s,
    new_start_s, new_end_s, ratio.
    """
    window_frames = int(SCAN_WINDOW_S / HOP_DURATION_S)
    min_frames = int(MIN_ANOMALY_DURATION_S / HOP_DURATION_S)
    anomalies = []

    old_frames = wp[:, 0]
    new_frames = wp[:, 1]

    i = 0
    while i + window_frames < len(wp):
        old_span = float(old_frames[i + window_frames] - old_frames[i])
        new_span = float(new_frames[i + window_frames] - new_frames[i])

        if old_span == 0:
            i += 1
            continue

        ratio = new_span / old_span

        if ratio < COMPRESS_RATIO or ratio > EXPAND_RATIO:
            # Walk forward to find the extent of the anomaly
            j = i + window_frames
            while j + 1 < len(wp):
                os = float(old_frames[j] - old_frames[i])
                ns = float(new_frames[j] - new_frames[i])
                if os == 0:
                    j += 1
                    continue
                r = ns / os
                if COMPRESS_RATIO <= r <= EXPAND_RATIO:
                    break
                j += 1

            span_frames = j - i
            if span_frames >= min_frames:
                kind = "compression" if ratio < COMPRESS_RATIO else "expansion"
                anomalies.append(
                    {
                        "type": kind,
                        "old_start_s": float(old_frames[i]) * HOP_DURATION_S,
                        "old_end_s": float(old_frames[j]) * HOP_DURATION_S,
                        "new_start_s": float(new_frames[i]) * HOP_DURATION_S,
                        "new_end_s": float(new_frames[j]) * HOP_DURATION_S,
                        "ratio": ratio,
                    }
                )
            i = j
        else:
            i += 1

    return anomalies


def load_timestamps(path: str) -> list[float]:
    """Load first-column timestamps from an Audacity label file.

    Returns a sorted list of unique timestamps.
    """
    timestamps: set[float] = set()
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if not parts:
                continue
            timestamps.add(float(parts[0]))
    return sorted(timestamps)


def snap_to_grid(t: float, grid: np.ndarray) -> float:
    """Snap timestamp *t* to the nearest point in *grid*."""
    idx = grid_index(t, grid)
    return float(grid[idx])


def grid_index(t: float, grid: np.ndarray) -> int:
    """Return the index of the nearest grid point to *t*."""
    pos = np.searchsorted(grid, t)
    if pos == 0:
        return 0
    if pos >= len(grid):
        return len(grid) - 1
    before = grid[pos - 1]
    after = grid[pos]
    if (t - before) <= (after - t):
        return pos - 1
    return pos


def find_chain_segments(
    labels: list[tuple[float, float, str]],
) -> list[list[int]]:
    """Split indexed labels into chain segments.

    A point label (start == end within POINT_LABEL_TOLERANCE) is always
    isolated (segment of length 1).

    Consecutive range labels where the end of one matches the start of the
    next within CHAIN_TOLERANCE form a chain segment.
    """
    n = len(labels)
    if n == 0:
        return []

    def is_point(i: int) -> bool:
        s, e, _ = labels[i]
        return abs(e - s) < POINT_LABEL_TOLERANCE

    segments: list[list[int]] = []
    current_chain: list[int] = []

    for i in range(n):
        if is_point(i):
            # Flush any pending chain
            if current_chain:
                segments.append(current_chain)
                current_chain = []
            segments.append([i])
            continue

        if not current_chain:
            current_chain = [i]
            continue

        prev_end = labels[current_chain[-1]][1]
        cur_start = labels[i][0]
        if abs(prev_end - cur_start) < CHAIN_TOLERANCE:
            current_chain.append(i)
        else:
            segments.append(current_chain)
            current_chain = [i]

    if current_chain:
        segments.append(current_chain)

    return segments


def _is_on_bar(t: float, bar_grid: np.ndarray | None) -> bool:
    """Return True if *t* snaps to a bar boundary within BAR_ALIGN_TOLERANCE."""
    if bar_grid is None or len(bar_grid) == 0:
        return False
    return abs(snap_to_grid(t, bar_grid) - t) < BAR_ALIGN_TOLERANCE


def _snap_to_bar_on_grid(
    warped_t: float, grid: np.ndarray, bar_grid: np.ndarray
) -> int:
    """Snap *warped_t* to the nearest bar boundary and return its index in *grid*."""
    bar_t = snap_to_grid(warped_t, bar_grid)
    return grid_index(bar_t, grid)


def compute_beat_counts(
    labels: list[tuple[float, float, str]],
    indices: list[int],
    old_beat_grid: np.ndarray,
) -> list[int]:
    """Compute beat counts for each label in a chain using the old beat grid.

    Returns a list of max(end_idx - start_idx, 1) per label.
    """
    counts = []
    for idx in indices:
        start, end, _ = labels[idx]
        si = grid_index(start, old_beat_grid)
        ei = grid_index(end, old_beat_grid)
        counts.append(max(ei - si, 1))
    return counts


def _remap_chain_preserving_beats(
    labels: list[tuple[float, float, str]],
    indices: list[int],
    warp: Callable[[float], float],
    grid: np.ndarray,
    old_beat_grid: np.ndarray,
    old_bar_grid: np.ndarray | None = None,
    new_bar_grid: np.ndarray | None = None,
) -> list[tuple[int, tuple[float, float, str]]]:
    """Remap a chain preserving original beat counts per label."""
    n = len(indices)
    beat_counts = compute_beat_counts(labels, indices, old_beat_grid)

    # DTW-warp the chain's first start to get cursor position on new grid
    first_start = labels[indices[0]][0]
    warped_first = warp(first_start)
    if _is_on_bar(first_start, old_bar_grid) and new_bar_grid is not None:
        cursor = _snap_to_bar_on_grid(warped_first, grid, new_bar_grid)
    else:
        cursor = grid_index(warped_first, grid)
    max_idx = len(grid) - 1

    result: list[tuple[int, tuple[float, float, str]]] = []
    for k in range(n):
        snap_start = cursor
        snap_end = cursor + beat_counts[k]

        # Clamp to grid bounds
        snap_start = min(snap_start, max_idx)
        snap_end = min(snap_end, max_idx)

        orig_idx = indices[k]
        lbl = labels[orig_idx][2]
        result.append((orig_idx, (float(grid[snap_start]), float(grid[snap_end]), lbl)))

        cursor = snap_end

        # Check drift against DTW for re-anchoring (except after last label)
        if k < n - 1:
            next_start = labels[indices[k + 1]][0]
            warped_next = warp(next_start)
            if _is_on_bar(next_start, old_bar_grid) and new_bar_grid is not None:
                dtw_gi = _snap_to_bar_on_grid(warped_next, grid, new_bar_grid)
            else:
                dtw_gi = grid_index(warped_next, grid)
            if abs(cursor - dtw_gi) > REANCHOR_THRESHOLD_BEATS:
                print(
                    f"  WARNING: re-anchoring after '{lbl}': "
                    f"cursor={cursor} vs dtw={dtw_gi} "
                    f"(drift={abs(cursor - dtw_gi)} beats)"
                )
                cursor = dtw_gi

    # Enforce exact chain: each label's end equals the next label's start
    for k in range(len(result) - 1):
        orig_k, (s_k, _e_k, l_k) = result[k]
        _orig_k1, (s_k1, e_k1, l_k1) = result[k + 1]
        result[k] = (orig_k, (s_k, s_k1, l_k))

    return result


def remap_chain_segment(
    labels: list[tuple[float, float, str]],
    indices: list[int],
    warp: Callable[[float], float],
    grid: np.ndarray,
    old_beat_grid: np.ndarray | None = None,
    old_bar_grid: np.ndarray | None = None,
    new_bar_grid: np.ndarray | None = None,
) -> list[tuple[int, tuple[float, float, str]]]:
    """Remap a chain of contiguous range labels.

    When old_beat_grid is provided, uses beat-count preservation.
    Otherwise uses DTW-snap fallback.

    Ensures monotonic grid indices, minimum 1 grid step per label, and exact
    chaining (each label's end equals the next label's start).

    Returns a list of (original_index, remapped_label) pairs.
    """
    if old_beat_grid is not None:
        return _remap_chain_preserving_beats(
            labels, indices, warp, grid, old_beat_grid,
            old_bar_grid=old_bar_grid, new_bar_grid=new_bar_grid,
        )

    n = len(indices)

    # Warp the chain boundary times and snap to grid indices
    boundary_old_times = [labels[indices[0]][0]]
    for idx in indices:
        boundary_old_times.append(labels[idx][1])

    raw_grid_indices = [grid_index(warp(t), grid) for t in boundary_old_times]

    # Enforce monotonic with minimum 1 step per label
    fixed = [raw_grid_indices[0]]
    for k in range(1, len(raw_grid_indices)):
        minimum = fixed[-1] + 1
        fixed.append(max(raw_grid_indices[k], minimum))

    # Clamp to grid bounds
    max_idx = len(grid) - 1
    for k in range(len(fixed)):
        if fixed[k] > max_idx:
            fixed[k] = max_idx

    result: list[tuple[int, tuple[float, float, str]]] = []
    for k in range(n):
        start_t = float(grid[fixed[k]])
        end_t = float(grid[fixed[k + 1]])
        orig_idx = indices[k]
        lbl = labels[orig_idx][2]
        result.append((orig_idx, (start_t, end_t, lbl)))

    return result


def parse_label_file(path: str) -> list[tuple[float, float, str]]:
    """Parse an Audacity label file (tab-separated: start, end, label)."""
    labels = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            start = float(parts[0])
            end = float(parts[1])
            label = parts[2]
            labels.append((start, end, label))
    return labels


def remap_labels(
    labels: list[tuple[float, float, str]],
    old_times: np.ndarray,
    new_times: np.ndarray,
    grid: np.ndarray,
    old_beat_grid: np.ndarray | None = None,
    old_bar_grid: np.ndarray | None = None,
    new_bar_grid: np.ndarray | None = None,
) -> list[tuple[float, float, str]]:
    """Warp label timestamps from old audio to new audio, snapped to grid."""

    def warp(t: float) -> float:
        return warp_timestamp(t, old_times, new_times)

    segments = find_chain_segments(labels)
    remapped: list[tuple[float, float, str]] = [("", "", "")] * len(labels)  # type: ignore[list-item]

    for seg in segments:
        if len(seg) == 1:
            idx = seg[0]
            start, end, lbl = labels[idx]
            is_point = abs(end - start) < POINT_LABEL_TOLERANCE
            new_start = snap_to_grid(warp(start), grid)
            if is_point:
                new_end = new_start
            else:
                new_end = snap_to_grid(warp(end), grid)
                if new_end <= new_start:
                    gi = grid_index(warp(start), grid)
                    new_end = float(grid[min(gi + 1, len(grid) - 1)])

            drift = abs(new_start - start)
            if drift > DRIFT_WARN_THRESHOLD_S:
                print(
                    f"  WARNING: label '{lbl}' drifted {drift:.2f}s "
                    f"(old={start:.3f} -> new={new_start:.3f})"
                )
            remapped[idx] = (new_start, new_end, lbl)
        else:
            chain_results = remap_chain_segment(
                labels, seg, warp, grid, old_beat_grid,
                old_bar_grid=old_bar_grid, new_bar_grid=new_bar_grid,
            )
            for orig_idx, label_tuple in chain_results:
                drift = abs(label_tuple[0] - labels[orig_idx][0])
                if drift > DRIFT_WARN_THRESHOLD_S:
                    print(
                        f"  WARNING: label '{label_tuple[2]}' drifted {drift:.2f}s "
                        f"(old={labels[orig_idx][0]:.3f} -> new={label_tuple[0]:.3f})"
                    )
                remapped[orig_idx] = label_tuple

    return remapped


def write_label_file(
    path: str, labels: list[tuple[float, float, str]]
) -> None:
    """Write labels in Audacity tab-separated format."""
    with open(path, "w") as f:
        for start, end, label in labels:
            f.write(f"{start:.6f}\t{end:.6f}\t{label}\n")


def remap_label_file(
    lf: str,
    old_times: np.ndarray,
    new_times: np.ndarray,
    grid: np.ndarray,
    out_path: Path,
    old_beat_grid: np.ndarray | None = None,
    use_bar_snap: bool = False,
    old_bar_grid: np.ndarray | None = None,
    bar_grid: np.ndarray | None = None,
) -> None:
    """Load, remap, and write a single label file."""
    lf_path = Path(lf)
    print(f"Remapping: {lf_path.name}")
    labels = parse_label_file(lf)
    # Beat-count preservation only when NOT bar-snapping
    old_grid_for_counts = None if use_bar_snap else old_beat_grid
    remapped = remap_labels(
        labels, old_times, new_times, grid, old_grid_for_counts,
        old_bar_grid=old_bar_grid, new_bar_grid=bar_grid,
    )
    dest = out_path / lf_path.name
    write_label_file(str(dest), remapped)
    print(f"  -> {dest}")


def run(
    old_audio: str,
    new_audio: str,
    label_files: list[str],
    outdir: str,
    new_beats_path: str,
    new_bars_path: str | None = None,
    bar_snap_files: list[str] | None = None,
    old_beats_path: str | None = None,
    old_bars_path: str | None = None,
) -> None:
    """Main pipeline: compute DTW, detect anomalies, remap all label files."""
    print(f"Loading old audio: {old_audio}")
    chroma_old = compute_chroma(old_audio)
    print(f"  {chroma_old.shape[1]} frames ({chroma_old.shape[1] * HOP_DURATION_S:.1f}s)")

    print(f"Loading new audio: {new_audio}")
    chroma_new = compute_chroma(new_audio)
    print(f"  {chroma_new.shape[1]} frames ({chroma_new.shape[1] * HOP_DURATION_S:.1f}s)")

    print("Computing DTW alignment...")
    wp = compute_dtw_path(chroma_old, chroma_new)
    print(f"  Warping path length: {len(wp)}")

    old_times, new_times = build_time_mapping(wp)

    print("Scanning for structural anomalies...")
    anomalies = detect_anomalies(wp)
    if anomalies:
        for a in anomalies:
            print(
                f"  {a['type'].upper()}: old [{a['old_start_s']:.1f}s - {a['old_end_s']:.1f}s] "
                f"-> new [{a['new_start_s']:.1f}s - {a['new_end_s']:.1f}s] "
                f"(ratio={a['ratio']:.2f})"
            )
    else:
        print("  No anomalies detected.")

    # Load grids
    beat_grid = np.array(load_timestamps(new_beats_path))
    print(f"Loaded beat grid: {len(beat_grid)} beats")

    bar_grid = None
    if new_bars_path:
        bar_grid = np.array(load_timestamps(new_bars_path))
        print(f"Loaded bar grid: {len(bar_grid)} bars")

    old_beat_grid = None
    if old_beats_path:
        old_beat_grid = np.array(load_timestamps(old_beats_path))
        print(f"Loaded old beat grid: {len(old_beat_grid)} beats")
        print("Beat-count preservation ON")

    old_bar_grid = None
    if old_bars_path:
        old_bar_grid = np.array(load_timestamps(old_bars_path))
        print(f"Loaded old bar grid: {len(old_bar_grid)} bars")
        print("Bar-boundary alignment ON")

    bar_snap_set = set(Path(p).name for p in (bar_snap_files or []))

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    for lf in label_files:
        lf_name = Path(lf).name
        use_bar_snap = lf_name in bar_snap_set
        if use_bar_snap:
            if bar_grid is None:
                print(
                    f"  ERROR: --bar-snap requested for '{lf_name}' "
                    "but no --new-bars provided"
                )
                sys.exit(1)
            grid = bar_grid
        else:
            grid = beat_grid
        remap_label_file(
            lf, old_times, new_times, grid, out_path,
            old_beat_grid=old_beat_grid, use_bar_snap=use_bar_snap,
            old_bar_grid=old_bar_grid, bar_grid=bar_grid,
        )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remap Audacity labels from old audio to new audio using DTW."
    )
    parser.add_argument("old_audio", help="Path to the old (reference) audio file")
    parser.add_argument("new_audio", help="Path to the new (target) audio file")
    parser.add_argument(
        "labels", nargs="+", help="One or more Audacity label files to remap"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="remapped/",
        help="Output directory for remapped label files (default: remapped/)",
    )
    parser.add_argument(
        "-b",
        "--new-beats",
        required=True,
        help="Audacity label file with new beat timestamps for snapping",
    )
    parser.add_argument(
        "-B",
        "--new-bars",
        default=None,
        help="Audacity label file with new bar timestamps for bar-level snapping",
    )
    parser.add_argument(
        "-s",
        "--bar-snap",
        action="append",
        default=[],
        help="Label file(s) to snap to bars instead of beats (repeatable)",
    )
    parser.add_argument(
        "-O",
        "--old-beats",
        default=None,
        help="Audacity label file with old beat timestamps (enables beat-count preservation)",
    )
    parser.add_argument(
        "-R",
        "--old-bars",
        default=None,
        help="Audacity label file with old bar timestamps (enables bar-boundary alignment for chain starts)",
    )
    args = parser.parse_args()

    run(
        args.old_audio,
        args.new_audio,
        args.labels,
        args.outdir,
        new_beats_path=args.new_beats,
        new_bars_path=args.new_bars,
        bar_snap_files=args.bar_snap,
        old_beats_path=args.old_beats,
        old_bars_path=args.old_bars,
    )
