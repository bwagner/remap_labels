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

AUDIO_EXTS = {".mp3", ".m4a", ".opus", ".wav", ".flac", ".ogg", ".aac", ".wma"}

# Below this bar-count ratio (min/max) auto-mode treats one audio as a
# subsequence of the other (clip/excerpt); at or above, both are full songs.
SUBSEQ_BAR_RATIO_THRESHOLD = 0.5


def resolve_subseq_mode(mode: str, old_bar_count: int, new_bar_count: int) -> bool:
    """Resolve 'auto'/'on'/'off' into the DTW subseq flag."""
    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode == "auto":
        smaller = min(old_bar_count, new_bar_count)
        larger = max(old_bar_count, new_bar_count)
        return smaller / larger < SUBSEQ_BAR_RATIO_THRESHOLD
    raise ValueError(f"invalid subseq mode: {mode!r} (expected auto/on/off)")


@dataclass
class DiscoveredInputs:
    audio: Path
    beats: Path
    bars: Path
    labels: list[Path] | None


def discover_dir_inputs(directory: Path, want_labels: bool) -> DiscoveredInputs:
    """Pick audio + beats_<stem>.txt + bars_<stem>.txt + (labels) from a dir.

    Labels are any *_<stem>.txt except beats_/bars_. want_labels=False returns labels=None.
    Raises ValueError on zero/multiple audio or missing beats/bars.
    """
    directory = Path(directory)
    audios = sorted(p for p in directory.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not audios:
        raise ValueError(f"no audio file in {directory}")
    if len(audios) > 1:
        names = ", ".join(p.name for p in audios)
        raise ValueError(f"multiple audio files in {directory}: {names}")
    audio = audios[0]
    stem = audio.stem

    beats = directory / f"beats_{stem}.txt"
    if not beats.is_file():
        raise ValueError(f"missing beats file: {beats.name} in {directory}")
    bars = directory / f"bars_{stem}.txt"
    if not bars.is_file():
        raise ValueError(f"missing bars file: {bars.name} in {directory}")

    labels: list[Path] | None = None
    if want_labels:
        labels = sorted(
            p for p in directory.iterdir()
            if p.suffix == ".txt"
            and p.name not in (beats.name, bars.name)
            and _matches_label_pattern(p.stem, stem)
        )
    return DiscoveredInputs(audio=audio, beats=beats, bars=bars, labels=labels)


def _matches_label_pattern(file_stem: str, audio_stem: str) -> bool:
    """True if file_stem is '<category>_<audio_stem>' with non-empty category."""
    prefix, sep, rest = file_stem.partition("_")
    return bool(sep) and bool(prefix) and rest == audio_stem


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


def compute_alignment(old_audio: str, new_audio: str, subseq: bool = True):
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

    print(f"Running DTW ({chroma_old.shape[1]} x {chroma_new.shape[1]} frames, subseq={subseq})...")
    _D, wp = librosa.sequence.dtw(X=chroma_old, Y=chroma_new, metric="cosine", subseq=subseq)
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

# Number of bars to sample for majority-vote bar shift detection
BAR_SHIFT_SAMPLE_SIZE = 20


CONFIDENT_VOTE_RATIO = 2.0


def choose_bar_shift(
    shifts: list[int],
) -> tuple[int, list[dict]]:
    """Choose bar shift from per-bar shift votes.

    Returns (best_shift, candidates) where each candidate is a dict
    with keys: shift, votes, bars (1-indexed), confident.

    Confident if the top candidate has >= CONFIDENT_VOTE_RATIO times
    the votes of the second candidate, or is the only candidate.
    """
    from collections import Counter

    counts = Counter(shifts)
    ranked = counts.most_common()

    candidates = []
    for shift, vote_count in ranked:
        bars = [i + 1 for i, s in enumerate(shifts) if s == shift]
        candidates.append({
            "shift": shift,
            "votes": vote_count,
            "bars": bars,
            "confident": False,  # set below
        })

    if len(candidates) == 1:
        candidates[0]["confident"] = True
    elif len(candidates) >= 2:
        top_votes = candidates[0]["votes"]
        second_votes = candidates[1]["votes"]
        candidates[0]["confident"] = top_votes >= second_votes * CONFIDENT_VOTE_RATIO

    return candidates[0]["shift"], candidates


def determine_bar_shift(
    old_bar_grid: list[float],
    new_bar_grid: list[float],
    warp,
) -> int:
    """Determine bar shift between old and new using majority vote.

    Warps multiple old bar positions through DTW and finds the most
    common shift. This is robust against DTW matching repeated
    intro riffs to the wrong repetition.
    """
    n = min(BAR_SHIFT_SAMPLE_SIZE, len(old_bar_grid))
    shifts = []
    for i in range(n):
        warped = warp(old_bar_grid[i])
        new_idx = grid_index(warped, new_bar_grid)
        shifts.append(new_idx - i)

    shift, candidates = choose_bar_shift(shifts)
    return shift


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


def _next_beat_time(
    abs_bar: int, whole_beat: int, bar_beats: list[int],
    beat_grid: list[float], bar_grid: list[float],
) -> float | None:
    """Time of the beat after (abs_bar, whole_beat): next beat in bar, or next bar's start."""
    if whole_beat + 1 < len(bar_beats):
        return beat_grid[bar_beats[whole_beat + 1]]
    if abs_bar + 1 < len(bar_grid):
        return bar_grid[abs_bar + 1]
    return None


def _fractional_beat_in_bar(
    t: float, bar_idx: int, beat_grid: list[float], bar_grid: list[float],
) -> Fraction:
    """Return fractional beat position (0-indexed) within a bar.

    If t falls on a beat, returns an integer Fraction (e.g. Fraction(2)).
    If t falls between beats, interpolates (e.g. Fraction(32, 100) for
    32% of the way from beat 0 to beat 1).
    """
    bar_beats = _beats_in_bar(bar_idx, beat_grid, bar_grid)
    if not bar_beats:
        return Fraction(0)

    # Check for exact beat match first
    for pos, beat_idx in enumerate(bar_beats):
        if abs(beat_grid[beat_idx] - t) < GRID_MATCH_TOLERANCE:
            return Fraction(pos)

    # Interpolate between beats
    for pos in range(len(bar_beats) - 1):
        beat_start = beat_grid[bar_beats[pos]]
        beat_end = beat_grid[bar_beats[pos + 1]]
        if beat_start <= t <= beat_end:
            frac = (t - beat_start) / (beat_end - beat_start)
            return Fraction(pos) + Fraction(frac).limit_denominator(1000)

    # After last beat in bar
    if t > beat_grid[bar_beats[-1]]:
        last_beat = beat_grid[bar_beats[-1]]
        if len(bar_beats) >= 2:
            beat_dur = beat_grid[bar_beats[-1]] - beat_grid[bar_beats[-2]]
            frac = (t - last_beat) / beat_dur
            return Fraction(len(bar_beats) - 1) + Fraction(frac).limit_denominator(1000)

    return Fraction(0)


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
        frac_in_bar = (entry.bar_offset - whole_bars) * beats_per_bar
        abs_bar = section_start_bar + whole_bars

        if entry.bar_offset < 0:
            # Pickup/anacrusis: place before bar_grid[section_start_bar]
            # using beat offset from bar 1
            bar1_time = bar_grid[section_start_bar]
            bar1_beats = _beats_in_bar(section_start_bar, beat_grid, bar_grid)
            if len(bar1_beats) >= 2:
                beat_dur = beat_grid[bar1_beats[1]] - beat_grid[bar1_beats[0]]
            else:
                beat_dur = (bar_grid[1] - bar_grid[0]) / beats_per_bar if len(bar_grid) > 1 else 0.5
            # bar_offset is negative, so this subtracts from bar 1
            start_time = bar1_time + float(entry.bar_offset) * beats_per_bar * beat_dur
        elif abs_bar >= len(bar_grid) or abs_bar >= section_end_bar:
            warnings.append(
                f"'{entry.label}' at bar {entry.bar_offset} dropped"
            )
            continue
        else:
            # Find beat within bar, interpolating for sub-beat positions
            bar_beats = _beats_in_bar(abs_bar, beat_grid, bar_grid)
            whole_beat = int(frac_in_bar)
            sub_beat_frac = float(frac_in_bar - whole_beat)

            if whole_beat < len(bar_beats):
                beat_time = beat_grid[bar_beats[whole_beat]]
                next_beat_time = _next_beat_time(
                    abs_bar, whole_beat, bar_beats, beat_grid, bar_grid,
                )
                if sub_beat_frac > 0 and next_beat_time is not None:
                    start_time = beat_time + sub_beat_frac * (next_beat_time - beat_time)
                else:
                    start_time = beat_time
            else:
                start_time = bar_grid[abs_bar]

        if entry.is_point:
            labels.append(LabelEntry(start_time, start_time, entry.label))
            continue

        # Compute end position: start offset + duration, both fractional
        end_bar_offset = entry.bar_offset + entry.bar_count

        if end_bar_offset < 0:
            # End in anacrusis (symmetric to start branch above)
            bar1_time = bar_grid[section_start_bar]
            bar1_beats = _beats_in_bar(section_start_bar, beat_grid, bar_grid)
            if len(bar1_beats) >= 2:
                beat_dur = beat_grid[bar1_beats[1]] - beat_grid[bar1_beats[0]]
            else:
                beat_dur = (bar_grid[1] - bar_grid[0]) / beats_per_bar if len(bar_grid) > 1 else 0.5
            end_time = bar1_time + float(end_bar_offset) * beats_per_bar * beat_dur
            labels.append(LabelEntry(start_time, end_time, entry.label))
            continue

        end_whole_bars = int(end_bar_offset)
        end_frac_in_bar = (end_bar_offset - end_whole_bars) * beats_per_bar
        end_abs_bar = section_start_bar + end_whole_bars

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
        elif end_frac_in_bar == 0:
            # Ends on a bar boundary
            end_time = bar_grid[end_abs_bar]
        else:
            end_whole_beat = int(end_frac_in_bar)
            end_sub_beat_frac = float(end_frac_in_bar - end_whole_beat)
            end_bar_beats = _beats_in_bar(end_abs_bar, beat_grid, bar_grid)
            if end_whole_beat < len(end_bar_beats):
                end_beat_time = beat_grid[end_bar_beats[end_whole_beat]]
                next_end_beat = _next_beat_time(
                    end_abs_bar, end_whole_beat, end_bar_beats, beat_grid, bar_grid,
                )
                if end_sub_beat_frac > 0 and next_end_beat is not None:
                    end_time = end_beat_time + end_sub_beat_frac * (next_end_beat - end_beat_time)
                else:
                    end_time = end_beat_time
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


def _time_to_bar_offset(
    t: float, beat_grid: list[float], bar_grid: list[float],
) -> Fraction:
    """Map an absolute time to a bar_offset (Fraction). Handles anacrusis (t < bar 1)."""
    if t < bar_grid[0] - GRID_MATCH_TOLERANCE:
        bar_beats = _beats_in_bar(0, beat_grid, bar_grid)
        beats_per_bar = max(len(bar_beats), 1)
        pickup_beats = [i for i, bt in enumerate(beat_grid) if bt < bar_grid[0] - GRID_MATCH_TOLERANCE]
        if pickup_beats:
            beats_before = len(pickup_beats)
            for pi, pbi in enumerate(pickup_beats):
                if pi + 1 < len(pickup_beats):
                    next_pbi = pickup_beats[pi + 1]
                    if beat_grid[pbi] <= t <= beat_grid[next_pbi]:
                        frac = (t - beat_grid[pbi]) / (beat_grid[next_pbi] - beat_grid[pbi])
                        beat_pos = -(beats_before - pi) + frac
                        return Fraction(beat_pos).limit_denominator(1000) / beats_per_bar
                else:
                    # Last pickup beat: interpolate toward bar 1
                    if beat_grid[pbi] <= t <= bar_grid[0]:
                        frac = (t - beat_grid[pbi]) / (bar_grid[0] - beat_grid[pbi])
                        beat_pos = -(beats_before - pi) + frac
                        return Fraction(beat_pos).limit_denominator(1000) / beats_per_bar
                    if abs(beat_grid[pbi] - t) < GRID_MATCH_TOLERANCE:
                        return Fraction(-(beats_before - pi)) / beats_per_bar
            # Before first pickup beat
            beat_pos = -beats_before - Fraction(
                int((beat_grid[pickup_beats[0]] - t) * 1000),
                int((beat_grid[1] - beat_grid[0]) * 1000),
            )
            return Fraction(beat_pos) / beats_per_bar
        # No pickup beats - estimate from bar spacing
        bar_dur = bar_grid[1] - bar_grid[0] if len(bar_grid) > 1 else 1.0
        return -Fraction(
            int((bar_grid[0] - t) * 1000),
            int(bar_dur * 1000),
        )
    chord_bar = _find_bar_for_time(t, bar_grid)
    frac_beat = _fractional_beat_in_bar(t, chord_bar, beat_grid, bar_grid)
    bar_beats = _beats_in_bar(chord_bar, beat_grid, bar_grid)
    beats_per_bar = max(len(bar_beats), 1)
    return Fraction(chord_bar) + frac_beat / beats_per_bar


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
    bar0_beats = max(len(_beats_in_bar(0, beat_grid, bar_grid)), 1)
    min_duration = Fraction(1, bar0_beats * 4)
    for label in labels:
        bar_offset = _time_to_bar_offset(label.start, beat_grid, bar_grid)
        if label.is_point:
            entries.append(SectionEntry(
                bar_offset=bar_offset,
                label=label.label,
                is_point=True,
                bar_count=Fraction(0),
            ))
        else:
            end_offset = _time_to_bar_offset(label.end, beat_grid, bar_grid)
            bar_count = max(end_offset - bar_offset, min_duration)
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


def main(
    old_audio: str,
    new_audio: str,
    new_beats_path: str,
    new_bars_path: str,
    old_beats_path: str,
    old_bars_path: str,
    label_files: list[str],
    outdir: str,
    subseq_mode: str = "auto",
) -> None:
    """Direct bar/beat remapping without sections.

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
    subseq = resolve_subseq_mode(subseq_mode, len(old_bar_grid), len(new_bar_grid))
    print(
        f"DTW subseq: {subseq} (mode={subseq_mode}, "
        f"bar counts old={len(old_bar_grid)} new={len(new_bar_grid)})"
    )
    old_times, new_times = compute_alignment(old_audio, new_audio, subseq=subseq)
    warp = make_warp_func(old_times, new_times)

    bar_shift = determine_bar_shift(old_bar_grid, new_bar_grid, warp)
    print(f"\nBar shift: {bar_shift:+d}")

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


def _resolve_cli_inputs(args) -> tuple[str, str, str, str, str, str, list[str], str]:
    """Resolve CLI args into (old_audio, new_audio, new_beats, new_bars,
    old_beats, old_bars, labels, outdir). Handles dir-mode auto-discovery."""
    old_is_dir = Path(args.old_audio).is_dir()
    new_is_dir = Path(args.new_audio).is_dir()
    if old_is_dir != new_is_dir:
        print(
            "Error: old_audio and new_audio must both be files or both be directories.",
            file=sys.stderr,
        )
        sys.exit(1)

    if old_is_dir:
        try:
            old = discover_dir_inputs(Path(args.old_audio), want_labels=True)
            new = discover_dir_inputs(Path(args.new_audio), want_labels=False)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        if not old.labels:
            print(
                f"Error: no label files found in {args.old_audio} "
                f"(expected <category>_{old.audio.stem}.txt).",
                file=sys.stderr,
            )
            sys.exit(1)
        extras = []
        for flag, val in [
            ("--old-beats", args.old_beats), ("--old-bars", args.old_bars),
            ("-b", args.new_beats), ("-B", args.new_bars),
        ]:
            if val is not None:
                extras.append((flag, val))
        if extras or args.labels:
            print(
                "Error: in dir-mode, do not pass --old-beats/--old-bars/-b/-B "
                "or positional label files; they are auto-discovered.",
                file=sys.stderr,
            )
            sys.exit(1)

        outdir = args.outdir if args.outdir else str(new.audio.parent)
        expanded = [
            "remap_labels.py",
            str(old.audio), str(new.audio),
            "--old-beats", str(old.beats),
            "--old-bars", str(old.bars),
            "-b", str(new.beats),
            "-B", str(new.bars),
            "-o", outdir,
            *(str(p) for p in old.labels),
        ]
        print("Dir-mode detected. Expanded CLI:")
        print("  " + " \\\n    ".join(expanded))
        print()
        return (
            str(old.audio), str(new.audio),
            str(new.beats), str(new.bars),
            str(old.beats), str(old.bars),
            [str(p) for p in old.labels],
            outdir,
        )

    missing = [
        name for name, val in [
            ("--old-beats", args.old_beats), ("--old-bars", args.old_bars),
            ("-b/--new-beats", args.new_beats), ("-B/--new-bars", args.new_bars),
        ] if val is None
    ]
    if missing:
        print(
            f"Error: file-mode requires: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.labels:
        print("Error: provide at least one label file to remap.", file=sys.stderr)
        sys.exit(1)

    outdir = args.outdir if args.outdir else str(Path(args.new_audio).parent)
    return (
        args.old_audio, args.new_audio,
        args.new_beats, args.new_bars,
        args.old_beats, args.old_bars,
        args.labels, outdir,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Remap Audacity labels from old audio to new audio. "
            "Pass two directories for auto-discovery mode."
        )
    )
    parser.add_argument(
        "-V", "--version", action="version",
        version=get_version_info(__version__),
    )
    parser.add_argument("old_audio", help="Original audio file or directory")
    parser.add_argument("new_audio", help="New (replacement) audio file or directory")
    parser.add_argument(
        "labels", nargs="*", help="Label .txt files to remap (file-mode only)"
    )
    parser.add_argument("--old-beats", help="Old beats file (file-mode only)")
    parser.add_argument("--old-bars", help="Old bars file (file-mode only)")
    parser.add_argument("-b", "--new-beats", help="New beats file (file-mode only)")
    parser.add_argument("-B", "--new-bars", help="New bars file (file-mode only)")
    parser.add_argument(
        "-o", "--outdir", default=None,
        help="Output directory (default: directory of new_audio)",
    )
    parser.add_argument(
        "--subseq", choices=["auto", "on", "off"], default="auto",
        help="DTW subseq mode: auto picks by bar-count ratio (default)",
    )
    args = parser.parse_args()

    resolved = _resolve_cli_inputs(args)
    main(*resolved[:2], *resolved[2:6], resolved[6], resolved[7], subseq_mode=args.subseq)
