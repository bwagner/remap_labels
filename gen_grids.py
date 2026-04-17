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


def _build_dbn_command(mp3_path: str, beats_per_bar: str | None = None) -> list[str]:
    """Construct the DBNDownBeatTracker command. --beats_per_bar must precede 'single'."""
    cmd = ["DBNDownBeatTracker"]
    if beats_per_bar:
        cmd += ["--beats_per_bar", beats_per_bar]
    cmd += ["single", str(mp3_path)]
    return cmd


def gen_grids(mp3_path: str, beats_per_bar: str | None = None) -> tuple[str, str]:
    """Run DBNDownBeatTracker and write beats/bars grid files.

    Returns (beats_path, bars_path).
    """
    mp3 = Path(mp3_path)
    songname = mp3.stem
    directory = mp3.parent
    beats_path = directory / f"beats_{songname}.txt"
    bars_path = directory / f"bars_{songname}.txt"

    cmd = _build_dbn_command(str(mp3), beats_per_bar=beats_per_bar)
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    beats_lines = []
    bars_lines = []
    bar_num = 0
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        timestamp = float(parts[0])
        beat_pos = int(parts[1])
        beats_lines.append(f"{timestamp:.6f}\n")
        if beat_pos == 1:
            bar_num += 1
            bars_lines.append(f"{timestamp:.6f}\t{timestamp:.6f}\t{bar_num}\n")

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
    parser.add_argument("mp3", help="Audio file (mp3, m4a, opus, wav, flac, ...)")
    parser.add_argument(
        "-b", "--beats-per-bar", default=None,
        help=(
            "Candidate bar lengths in beats (comma-separated list). "
            "Forwarded to DBNDownBeatTracker --beats_per_bar. "
            "Examples: 5 (force 5/4), 3,4 (DBN chooses, its default), 4,5."
        ),
    )
    args = parser.parse_args()

    mp3 = Path(args.mp3)
    if not mp3.exists():
        print(f"Error: file not found: {mp3}", file=sys.stderr)
        sys.exit(1)

    gen_grids(str(mp3), beats_per_bar=args.beats_per_bar)
