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
) -> list[tuple[float, float, str]]:
    """Warp label timestamps from old audio to new audio."""
    remapped = []
    for start, end, label in labels:
        new_start = warp_timestamp(start, old_times, new_times)
        is_point = start == end
        if is_point:
            new_end = new_start
        else:
            new_end = warp_timestamp(end, old_times, new_times)

        drift = abs(new_start - start)
        if drift > DRIFT_WARN_THRESHOLD_S:
            print(
                f"  WARNING: label '{label}' drifted {drift:.2f}s "
                f"(old={start:.3f} -> new={new_start:.3f})"
            )

        remapped.append((new_start, new_end, label))
    return remapped


def write_label_file(
    path: str, labels: list[tuple[float, float, str]]
) -> None:
    """Write labels in Audacity tab-separated format."""
    with open(path, "w") as f:
        for start, end, label in labels:
            f.write(f"{start:.6f}\t{end:.6f}\t{label}\n")


def run(
    old_audio: str,
    new_audio: str,
    label_files: list[str],
    outdir: str,
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

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    for lf in label_files:
        lf_path = Path(lf)
        print(f"Remapping: {lf_path.name}")
        labels = parse_label_file(lf)
        remapped = remap_labels(labels, old_times, new_times)
        dest = out_path / lf_path.name
        write_label_file(str(dest), remapped)
        print(f"  -> {dest}")

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
    args = parser.parse_args()

    run(args.old_audio, args.new_audio, args.labels, args.outdir)
