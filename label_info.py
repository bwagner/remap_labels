#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Display Audacity label tracks with bar/beat positions and detect repeating patterns.

Usage:
    # Auto-discover from mp3/aup3:
    label_info.py song.mp3

    # Explicit label files + grids:
    label_info.py -l chords.txt parts.txt --bars bars.txt --beats beats.txt

    # Interleave multiple tracks:
    label_info.py song.mp3 --interleave
"""

import sys
from pathlib import Path

from remap_labels import (
    LabelEntry,
    _beat_position_in_bar,
    _find_bar_for_time,
    load_labels,
    load_timestamps,
)


def discover_files(audio_path: str) -> dict:
    """Discover label, bars, and beats files from an mp3/aup3 path.

    Convention: given ``<dir>/<songname>.mp3``, looks for:
    - ``bars_<songname>.txt`` and ``beats_<songname>.txt``
    - ``<prefix>_<songname>.txt`` for any prefix (label tracks)
    """
    p = Path(audio_path)
    directory = p.parent
    songname = p.stem
    suffix = f"_{songname}.txt"

    bars_path = directory / f"bars{suffix}"
    if not bars_path.exists():
        raise FileNotFoundError(f"bars file not found: {bars_path}")

    beats_path = directory / f"beats{suffix}"
    if not beats_path.exists():
        raise FileNotFoundError(f"beats file not found: {beats_path}")

    reserved = {f"bars{suffix}", f"beats{suffix}"}
    labels = []
    for f in sorted(directory.iterdir()):
        if f.name in reserved:
            continue
        if f.name.endswith(suffix) and f.name != f"{songname}.txt":
            labels.append(str(f))

    if not labels:
        raise FileNotFoundError(
            f"No label files matching *{suffix} found in {directory}"
        )

    return {
        "bars": str(bars_path),
        "beats": str(beats_path),
        "labels": labels,
    }


def prefix_from_filename(filename: str, songname: str) -> str:
    """Extract the prefix from a label filename.

    ``chords_my_song.txt`` with songname ``my_song`` returns ``chords``.
    """
    name = Path(filename).stem
    suffix = f"_{songname}"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def label_to_bar_beat(
    t: float,
    bar_grid: list[float],
    beat_grid: list[float],
    *,
    show_beats: bool = False,
) -> str:
    """Convert a timestamp to a bar.beat string.

    Returns ``bar N`` if on a bar boundary (unless show_beats is True),
    or ``bar N.B`` if on a sub-bar beat.
    """
    bar_idx = _find_bar_for_time(t, bar_grid)
    beat_pos = _beat_position_in_bar(t, bar_idx, beat_grid, bar_grid)
    bar_num = bar_idx + 1

    if beat_pos == 0 and not show_beats:
        return f"bar {bar_num}"
    beat_num = beat_pos + 1 if beat_pos > 0 else 1
    return f"bar {bar_num}.{beat_num}"


def format_track(
    labels: list[LabelEntry],
    bar_grid: list[float],
    beat_grid: list[float],
    *,
    show_beats: bool = False,
) -> list[str]:
    """Format a label track as lines with bar/beat positions."""
    lines = []
    for lbl in labels:
        pos = label_to_bar_beat(lbl.start, bar_grid, beat_grid, show_beats=show_beats)
        lines.append(f"{pos}: {lbl.label}")
    return lines


def interleave_tracks(
    tracks: dict[str, list[LabelEntry]],
    bar_grid: list[float],
    beat_grid: list[float],
    *,
    show_beats: bool = False,
    show_prefix: bool = False,
) -> list[str]:
    """Interleave multiple label tracks, merging coinciding labels.

    Labels at the same bar.beat position are joined with space.
    With show_prefix, each label gets a ``(prefix)`` suffix.
    """
    # Collect all (bar_beat_str, sort_key, prefix, label_text)
    items: list[tuple[str, float, str, str]] = []
    for prefix, labels in tracks.items():
        for lbl in labels:
            pos = label_to_bar_beat(
                lbl.start, bar_grid, beat_grid, show_beats=show_beats
            )
            items.append((pos, lbl.start, prefix, lbl.label))

    # Sort by timestamp
    items.sort(key=lambda x: x[1])

    # Group by position string
    lines = []
    current_pos = None
    current_parts: list[str] = []
    for pos, _t, prefix, text in items:
        if pos != current_pos:
            if current_pos is not None:
                lines.append(f"{current_pos}: {' '.join(current_parts)}")
            current_pos = pos
            current_parts = []
        if show_prefix:
            current_parts.append(f"{text} ({prefix})")
        else:
            current_parts.append(text)

    if current_pos is not None:
        lines.append(f"{current_pos}: {' '.join(current_parts)}")

    return lines


def _build_bars_content(
    labels: list[LabelEntry],
    bar_grid: list[float],
) -> tuple[list[tuple[str, ...]], int]:
    """Build per-bar label sequences. Returns (bars_content, first_bar_index)."""
    bar_labels: list[tuple[int, str]] = []
    for lbl in labels:
        bar_idx = _find_bar_for_time(lbl.start, bar_grid)
        bar_labels.append((bar_idx, lbl.label))

    if not bar_labels:
        return [], 0

    min_bar = bar_labels[0][0]
    bar_labels = [(b - min_bar, lbl) for b, lbl in bar_labels]

    max_bar = max(b for b, _ in bar_labels)
    bars_content: list[tuple[str, ...]] = []
    for bar_idx in range(max_bar + 1):
        bar_items = tuple(lbl for b, lbl in bar_labels if b == bar_idx)
        bars_content.append(bar_items)

    return bars_content, min_bar


def _find_repeat_at(bars_content: list[tuple[str, ...]], start: int) -> dict | None:
    """Find the smallest repeating pattern starting at position start.

    Returns a dict with keys: bars, count, bar_tuples, or None if no repeat.
    """
    n = len(bars_content)
    remaining = n - start
    for pat_len in range(1, remaining // 2 + 1):
        pattern = bars_content[start : start + pat_len]
        count = 1
        pos = start + pat_len
        while pos + pat_len <= n:
            if bars_content[pos : pos + pat_len] == pattern:
                count += 1
                pos += pat_len
            else:
                break
        if count >= 2:
            flat_labels = []
            for bar_tuple in pattern:
                flat_labels.extend(bar_tuple)
            return {
                "labels": flat_labels,
                "bars": pat_len,
                "count": count,
                "bar_tuples": list(pattern),
            }
    return None


def detect_pattern(
    labels: list[LabelEntry],
    bar_grid: list[float],
    beat_grid: list[float],
) -> list[dict]:
    """Detect repeating patterns in a label track.

    Returns a list of segment dicts. Each segment is either:
    - A repeat: {type: "repeat", bars, count, labels, bar_tuples, start_bar, end_bar}
    - A literal: {type: "literal", start_bar, end_bar, bar_tuples}
    """
    if not labels:
        return []

    bars_content, first_bar = _build_bars_content(labels, bar_grid)
    n_bars = len(bars_content)
    segments = []
    pos = 0

    while pos < n_bars:
        rep = _find_repeat_at(bars_content, pos)
        if rep and any(t for t in rep["bar_tuples"] if t):
            segments.append({
                "type": "repeat",
                "bars": rep["bars"],
                "count": rep["count"],
                "labels": rep["labels"],
                "bar_tuples": rep["bar_tuples"],
                "start_bar": first_bar + pos,
                "end_bar": first_bar + pos + rep["bars"] * rep["count"],
            })
            pos += rep["bars"] * rep["count"]
        else:
            # Single non-repeating bar - skip if empty
            if bars_content[pos]:
                segments.append({
                    "type": "literal",
                    "start_bar": first_bar + pos,
                    "end_bar": first_bar + pos + 1,
                    "bar_tuples": [bars_content[pos]],
                })
            pos += 1

    return segments


def _primary_track(
    tracks: dict[str, list[LabelEntry]],
    primary: str | None = None,
) -> tuple[str, list[LabelEntry]]:
    """Return the primary track for pattern detection.

    If primary is given, use that track name. Otherwise pick the track
    with the most labels.
    """
    if primary is not None:
        if primary not in tracks:
            raise ValueError(
                f"'{primary}' not found in tracks: {list(tracks.keys())}"
            )
        return primary, tracks[primary]
    return max(tracks.items(), key=lambda kv: len(kv[1]))


def _print_interleave_compact(
    tracks: dict[str, list[LabelEntry]],
    bar_grid: list[float],
    beat_grid: list[float],
    *,
    show_beats: bool = False,
    show_prefix: bool = False,
    primary: str | None = None,
) -> None:
    """Print interleaved tracks compactly.

    Detects patterns on the primary track, then overlays other tracks'
    labels. Bars with extra labels are spelled out as literals.
    """
    primary_name, primary_labels = _primary_track(tracks, primary=primary)
    segments = detect_pattern(primary_labels, bar_grid, beat_grid)

    # Build interleaved lines indexed by bar number
    all_lines = interleave_tracks(
        tracks, bar_grid, beat_grid,
        show_beats=show_beats, show_prefix=show_prefix,
    )
    bar_to_lines: dict[int, list[str]] = {}
    for line in all_lines:
        bar_num = int(line.split(":")[0].replace("bar ", "").split(".")[0])
        bar_to_lines.setdefault(bar_num, []).append(line)

    # Collect bars that have labels from sparse non-primary tracks
    # (tracks with fewer labels than primary don't repeat every bar,
    # so their labels mark structural boundaries worth splitting at)
    primary_count = len(primary_labels)
    other_labels = {}
    omitted_tracks = []
    for name, labels in tracks.items():
        if name == primary_name:
            continue
        if len(labels) >= primary_count:
            omitted_tracks.append(name)
            continue
        for lbl in labels:
            bar_idx = _find_bar_for_time(lbl.start, bar_grid)
            other_labels.setdefault(bar_idx, []).append(lbl)

    for seg in segments:
        if seg["type"] == "literal":
            bar_num = seg["start_bar"] + 1
            for line in bar_to_lines.get(bar_num, []):
                print(f"  {line}")
            continue

        # Repeat segment: split at bars that have other-track labels
        start = seg["start_bar"]
        end = seg["end_bar"]
        pat_len = seg["bars"]

        # Find bars within this repeat that have other-track labels
        extra_bars = sorted(
            b for b in range(start, end) if b in other_labels
        )

        if not extra_bars:
            pat_str = _format_pattern(seg["bar_tuples"], seg["count"], seg["bars"])
            print(f"  bars {start + 1}-{end}: {pat_str}")
            continue

        # Split repeat at extra-label bars
        # Collect split points (bar indices where a new sub-range starts)
        splits = [start] + extra_bars
        # Remove duplicates and sort
        splits = sorted(set(splits))

        for i, sp in enumerate(splits):
            sp_end = splits[i + 1] if i + 1 < len(splits) else end

            # Annotation from other-track labels at this split point
            annotation = ""
            if sp in other_labels:
                texts = [lbl.label for lbl in other_labels[sp]]
                annotation = f"  [{', '.join(texts)}]"

            n_bars_range = sp_end - sp
            on_boundary = (sp - start) % pat_len == 0
            n_reps = n_bars_range // pat_len if on_boundary else 0

            if n_reps >= 2:
                rep_end = sp + n_reps * pat_len
                pat_str = _format_pattern(seg["bar_tuples"], n_reps, pat_len)
                print(f"  bars {sp + 1}-{rep_end}: {pat_str}{annotation}")
                # Spell out leftover bars
                pos = rep_end
                while pos < sp_end:
                    bar_num = pos + 1
                    for line in bar_to_lines.get(bar_num, []):
                        print(f"  {line}")
                    pos += 1
            elif n_reps == 1:
                # Single pattern instance - spell out with annotation
                bar_num = sp + 1
                for line in bar_to_lines.get(bar_num, []):
                    print(f"  {line}")
                pos = sp + 1
                while pos < sp_end:
                    bar_num = pos + 1
                    for line in bar_to_lines.get(bar_num, []):
                        print(f"  {line}")
                    pos += 1
            else:
                # Not on boundary - spell out
                pos = sp
                while pos < sp_end:
                    bar_num = pos + 1
                    for line in bar_to_lines.get(bar_num, []):
                        print(f"  {line}")
                    pos += 1

    if omitted_tracks:
        names = ", ".join(omitted_tracks)
        print(f"\n  ({names} omitted from compact view, use -s to see separately)")


def _format_pattern(bar_tuples: list[tuple[str, ...]], count: int, bars: int) -> str:
    """Format a repeating pattern as | bar1 | bar2 | xN."""
    bar_strs = []
    for bar_tuple in bar_tuples:
        bar_strs.append(", ".join(bar_tuple))
    label_str = " | ".join(bar_strs)
    return f"| {label_str} | x{count}"


def _print_compact(
    formatted_lines: list[str],
    labels: list[LabelEntry],
    bar_grid: list[float],
    beat_grid: list[float],
) -> None:
    """Print labels compactly: repeating patterns collapsed, literals spelled out."""
    segments = detect_pattern(labels, bar_grid, beat_grid)

    # Build a bar->lines index for picking out literal bars
    bar_to_lines: dict[int, list[str]] = {}
    for line in formatted_lines:
        bar_num = int(line.split(":")[0].replace("bar ", "").split(".")[0])
        bar_to_lines.setdefault(bar_num, []).append(line)

    for seg in segments:
        if seg["type"] == "repeat":
            start = seg["start_bar"] + 1  # 1-indexed
            end = seg["end_bar"]  # 1-indexed (exclusive -> last bar)
            pat_str = _format_pattern(seg["bar_tuples"], seg["count"], seg["bars"])
            print(f"  bars {start}-{end}: {pat_str}")
        else:
            bar_num = seg["start_bar"] + 1  # 1-indexed
            for line in bar_to_lines.get(bar_num, []):
                print(f"  {line}")


def print_track(
    name: str,
    labels: list[LabelEntry],
    bar_grid: list[float],
    beat_grid: list[float],
    *,
    show_beats: bool = False,
    expand: bool = False,
) -> None:
    """Print a single label track with bar/beat positions and pattern info.

    With expand=False (default), repeating patterns are shown compactly.
    With expand=True, every label is spelled out.
    """
    print(f"\n=== {name} ===")
    lines = format_track(labels, bar_grid, beat_grid, show_beats=show_beats)

    if expand:
        for line in lines:
            print(f"  {line}")
    else:
        _print_compact(lines, labels, bar_grid, beat_grid)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Display Audacity label tracks with bar/beat positions."
    )
    parser.add_argument(
        "source", nargs="?",
        help="Audio file (.mp3/.aup3) for auto-discovery",
    )
    parser.add_argument(
        "-l", "--labels", nargs="+",
        help="Label files (overrides auto-discovery)",
    )
    parser.add_argument(
        "--bars",
        help="Bars grid file (overrides auto-discovery)",
    )
    parser.add_argument(
        "--beats",
        help="Beats grid file (overrides auto-discovery)",
    )
    parser.add_argument(
        "-s", "--separate", action="store_true",
        help="Show each track separately instead of interleaved",
    )
    parser.add_argument(
        "--show-beats", action="store_true",
        help="Always show beat number, even on bar boundaries",
    )
    parser.add_argument(
        "-x", "--show-prefix", action="store_true",
        help="Show track prefix on each label",
    )
    parser.add_argument(
        "-e", "--expand", action="store_true",
        help="Spell out every label instead of collapsing repeating patterns",
    )
    parser.add_argument(
        "-p", "--primary",
        help="Track to use for pattern detection (default: most labels)",
    )
    args = parser.parse_args()

    # Resolve inputs
    if args.source:
        discovered = discover_files(args.source)
        bars_path = args.bars or discovered["bars"]
        beats_path = args.beats or discovered["beats"]
        label_paths = args.labels or discovered["labels"]
        songname = Path(args.source).stem
    elif args.labels:
        if not args.bars or not args.beats:
            print("Error: --bars and --beats required when using -l.", file=sys.stderr)
            sys.exit(1)
        bars_path = args.bars
        beats_path = args.beats
        label_paths = args.labels
        songname = None
    else:
        print("Error: provide a source audio file or -l with label files.", file=sys.stderr)
        sys.exit(1)

    bar_grid = load_timestamps(bars_path)
    beat_grid = load_timestamps(beats_path)

    # Load all tracks
    tracks: dict[str, list[LabelEntry]] = {}
    for lp in label_paths:
        if songname:
            prefix = prefix_from_filename(Path(lp).name, songname)
        else:
            prefix = Path(lp).stem
        tracks[prefix] = load_labels(lp)

    if args.separate:
        for name, labels in tracks.items():
            print_track(
                name, labels, bar_grid, beat_grid,
                show_beats=args.show_beats, expand=args.expand,
            )
    else:
        if args.expand:
            lines = interleave_tracks(
                tracks, bar_grid, beat_grid,
                show_beats=args.show_beats, show_prefix=args.show_prefix,
            )
            for line in lines:
                print(f"  {line}")
        else:
            _print_interleave_compact(
                tracks, bar_grid, beat_grid,
                show_beats=args.show_beats, show_prefix=args.show_prefix,
                primary=args.primary,
            )


if __name__ == "__main__":
    main()
