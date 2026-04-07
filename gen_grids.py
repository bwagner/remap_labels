#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate beats and bars grid files from an mp3 using DBNDownBeatTracker.

Usage:
    gen_grids.py song.mp3

Creates beats_<songname>.txt and bars_<songname>.txt in the same directory.
"""

import subprocess
import sys
from pathlib import Path


def gen_grids(mp3_path: str) -> tuple[str, str]:
    """Run DBNDownBeatTracker and write beats/bars grid files.

    Returns (beats_path, bars_path).
    """
    mp3 = Path(mp3_path)
    songname = mp3.stem
    directory = mp3.parent
    beats_path = directory / f"beats_{songname}.txt"
    bars_path = directory / f"bars_{songname}.txt"

    print(f"Running DBNDownBeatTracker on {mp3}...")
    result = subprocess.run(
        ["DBNDownBeatTracker", "single", str(mp3)],
        capture_output=True, text=True, check=True,
    )

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
        description="Generate beats and bars grid files from an mp3.",
    )
    parser.add_argument("mp3", help="Audio file (.mp3)")
    args = parser.parse_args()

    mp3 = Path(args.mp3)
    if not mp3.exists():
        print(f"Error: file not found: {mp3}", file=sys.stderr)
        sys.exit(1)

    gen_grids(str(mp3))
