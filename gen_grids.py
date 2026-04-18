#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate beats and bars grid files from an audio file using DBNDownBeatTracker.

Accepts any audio format madmom/ffmpeg can decode (mp3, m4a, opus, wav, flac, ...).

Usage:
    gen_grids.py song.mp3
    gen_grids.py song.m4a
    gen_grids.py --beats-per-bar 5 mission_impossible.m4a
    gen_grids.py --beats-per-bar 3,4 waltz_or_common.mp3

Creates beats_<songname>.txt and bars_<songname>.txt in the same directory.

The --beats-per-bar value is forwarded verbatim to DBNDownBeatTracker's
--beats_per_bar (comma-separated list of candidate bar lengths in beats).
DBN's default is 3,4. Pass a single value to force a time signature, or a
list to let DBN choose among candidates.
"""

import subprocess
import sys
from pathlib import Path

AUDIO_EXTS = {".mp3", ".m4a", ".opus", ".wav", ".flac", ".ogg", ".aac", ".wma"}


def _discover_audio_file(directory: Path) -> Path:
    """Find the unique audio file in a directory. Raises ValueError on 0 or >1."""
    matches = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in AUDIO_EXTS
    )
    if not matches:
        raise ValueError(f"no audio file in {directory}")
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise ValueError(f"multiple audio files in {directory}: {names}")
    return matches[0]


def _build_dbn_command(
    mp3_path: str,
    beats_per_bar: str | None = None,
    min_bpm: str | None = None,
    max_bpm: str | None = None,
) -> list[str]:
    """Construct the DBNDownBeatTracker command. All DBN options must precede 'single'."""
    cmd = ["DBNDownBeatTracker"]
    if beats_per_bar:
        cmd += ["--beats_per_bar", beats_per_bar]
    if min_bpm:
        cmd += ["--min_bpm", min_bpm]
    if max_bpm:
        cmd += ["--max_bpm", max_bpm]
    cmd += ["single", str(mp3_path)]
    return cmd


def _format_bars(downbeats: list[float], span: bool = False) -> list[str]:
    """Format downbeat timestamps as bars grid lines.

    Event mode (default): each downbeat as a zero-duration label.
    Span mode: each bar spans from its downbeat to the next; last bar omitted
    (end time unknown). Matches beats2bars.py convention.
    """
    if span:
        return [
            f"{downbeats[i]:.6f}\t{downbeats[i+1]:.6f}\t{i+1}\n"
            for i in range(len(downbeats) - 1)
        ]
    return [f"{t:.6f}\t{t:.6f}\t{i+1}\n" for i, t in enumerate(downbeats)]


def gen_grids(
    mp3_path: str,
    beats_per_bar: str | None = None,
    span: bool = False,
    min_bpm: str | None = None,
    max_bpm: str | None = None,
) -> tuple[str, str]:
    """Run DBNDownBeatTracker and write beats/bars grid files.

    Returns (beats_path, bars_path).
    """
    mp3 = Path(mp3_path)
    songname = mp3.stem
    directory = mp3.parent
    beats_path = directory / f"beats_{songname}.txt"
    bars_path = directory / f"bars_{songname}.txt"

    cmd = _build_dbn_command(
        str(mp3), beats_per_bar=beats_per_bar, min_bpm=min_bpm, max_bpm=max_bpm,
    )
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    beats_lines = []
    downbeats = []
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        timestamp = float(parts[0])
        beat_pos = int(parts[1])
        beats_lines.append(f"{timestamp:.6f}\n")
        if beat_pos == 1:
            downbeats.append(timestamp)

    bars_lines = _format_bars(downbeats, span=span)
    beats_path.write_text("".join(beats_lines))
    bars_path.write_text("".join(bars_lines))

    print("Created:")
    print(f"  {beats_path} ({len(beats_lines)} beats)")
    print(f"  {bars_path} ({len(bars_lines)} bars)")

    return str(beats_path), str(bars_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate beats and bars grid files from an audio file.",
    )
    parser.add_argument(
        "mp3",
        help="Audio file (mp3, m4a, opus, wav, flac, ...) or directory containing one",
    )
    parser.add_argument(
        "-b", "--beats-per-bar", default=None,
        help=(
            "Candidate bar lengths in beats (comma-separated list). "
            "Forwarded to DBNDownBeatTracker --beats_per_bar. "
            "Examples: 5 (force 5/4), 3,4 (DBN chooses, its default), 4,5."
        ),
    )
    parser.add_argument(
        "-s", "--span", action="store_true",
        help="Emit duration labels (start...end). Default: zero-duration events at each downbeat.",
    )
    parser.add_argument(
        "--min-bpm", default=None,
        help="Minimum tempo for DBN (forwarded to --min_bpm). Default: DBN's 55.",
    )
    parser.add_argument(
        "--max-bpm", default=None,
        help="Maximum tempo for DBN (forwarded to --max_bpm). Default: DBN's 215.",
    )
    args = parser.parse_args()

    mp3 = Path(args.mp3)
    if not mp3.exists():
        print(f"Error: not found: {mp3}", file=sys.stderr)
        sys.exit(1)

    if mp3.is_dir():
        try:
            mp3 = _discover_audio_file(mp3)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Dir-mode: discovered audio {mp3}")

    gen_grids(
        str(mp3),
        beats_per_bar=args.beats_per_bar,
        span=args.span,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
    )
