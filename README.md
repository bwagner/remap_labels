# remap_labels

Remap Audacity label tracks (chords, parts, guitar, rehearsal marks, etc.) from an old audio version to a new one.

When a musical director swaps out audio files - different tempo, rearranged sections, different version - this tool reconstructs your meticulous labeling on the new audio using musical structure (bars, beats, sections) rather than raw timestamps.

## Installation

### uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### madmom (provides DBNDownBeatTracker)

```bash
uv tool install madmom
```

### Audacity

Download from [audacityteam.org](https://www.audacityteam.org/) or install via your package manager.

### remap_labels

Add the project directory to your PATH:

```bash
# In ~/.bashrc or ~/.bash_profile
export PATH="$HOME/projects/remap_labels:$PATH"
```

Make the script executable (one time):

```bash
chmod +x ~/projects/remap_labels/remap_labels.py
```

The script uses a uv shebang and downloads its Python dependencies (librosa, numpy, etc.) automatically on first run.

## Workflow

The process has three stages: generate new beat/bar grids, run the remapper, import into Audacity.

### 1. Generate beats and bars for the new audio

```bash
gen_grids.py new_audio.mp3
```

Runs DBNDownBeatTracker and creates `beats_<songname>.txt` and `bars_<songname>.txt` in the same directory as the mp3.

### 2. Run remap_labels

```bash
remap_labels.py \
    old_audio.mp3 \
    new_audio.mp3 \
    --old-beats beats_old.txt \
    --old-bars bars_old.txt \
    -b new_beats.txt \
    -B new_bars.txt \
    chords_old.txt parts_old.txt guit_old.txt rehearse_old.txt
```

(Assumes `remap_labels.py` is on your PATH - see Installation.)

All label files listed at the end are remapped. Output goes to `remapped/` (override with `-o`).

#### Arguments

| Arg | Description |
|-----|-------------|
| `old_audio` | Original audio file |
| `new_audio` | New (replacement) audio file |
| `--old-beats` | Old beat grid |
| `--old-bars` | Old bar grid |
| `-b` / `--new-beats` | New beat grid (from DBNDownBeatTracker) |
| `-B` / `--new-bars` | New bar grid (downbeats from DBNDownBeatTracker) |
| `-o` / `--outdir` | Output directory (default: `remapped/`) |
| label files... | Any number of label files to remap |

### 3. Import into Audacity

1. Open the new audio file in Audacity.
2. File > Import > Labels for each file in `remapped/`:
   - `chords_*.txt`
   - `parts_*.txt`
   - `guit_*.txt` (if present)
   - `review.txt` (flags places needing manual attention)
3. Also import `new_beats.txt` and `new_bars.txt` as label tracks for reference.
4. Check the review track for issues, fix manually.

## What the tool does

1. **Parses old labels** into absolute (bar, beat) positions using the old bar/beat grids. Every label track is treated the same - no special handling.
2. **Aligns old and new audio** using chroma-based Dynamic Time Warping (see [docs/dtw_chroma.md](docs/dtw_chroma.md)) to find where bar 1 starts in the new version.
3. **Reconstructs labels**: each label is placed at its original (bar, beat) position on the new grid. A chord at bar 5, beat 3 in the old goes to bar 5, beat 3 in the new.
4. **Generates a review track** flagging:
   - Structural anomalies (sections added/removed between old and new)
   - Dropped labels (bar doesn't exist in new grid)

## Example

```bash
cd blues_brothers_everybody_needs_somebody/

# Step 1: new beats/bars
gen_grids.py 21_Everybody_needs_somebody_Playback_band.mp3

# Step 2: remap
remap_labels.py \
    blues_brothers_everybody_needs_somebody.mp3 \
    21_Everybody_needs_somebody_Playback_band.mp3 \
    --old-beats beats_blues_brothers_everybody_needs_somebody.txt \
    --old-bars bars_blues_brothers_everybody_needs_somebody.txt \
    -b new_beats.txt \
    -B new_bars.txt \
    chords_blues_brothers_everybody_needs_somebody.txt \
    parts_blues_brothers_everybody_needs_somebody.txt \
    guit_blues_brothers_everybody_needs_somebody.txt

# Step 3: import remapped/*.txt into Audacity with the new audio
```

## label_info - inspect label tracks

Display label tracks with bar/beat positions instead of timestamps. Detects repeating chord patterns and interleaves multiple tracks.

```bash
# Auto-discover all label/grid files from mp3 or aup3:
label_info.py song.mp3

# Separate per-track view:
label_info.py song.mp3 -s

# Show track prefixes:
label_info.py song.mp3 -p

# Spell out every label (no pattern collapsing):
label_info.py song.mp3 -e

# Explicit files:
label_info.py -l chords.txt parts.txt --bars bars.txt --beats beats.txt
```

### Example output

```
  bars 1-8: | C, F | Bb, F | x4  [intro_instr]
  bars 9-30: | C, F | Bb, F | x11  [intro_speech]
  bars 31-56: | C, F | Bb, F | x13  [verse1]
  bar 57: Am bridge1
  bar 59: F
  bar 61: Am
  bar 63: G
  bars 65-80: | C | F | x8  [verse2_instr]
```

Bars with labels from multiple tracks are merged. Repeating chord patterns are collapsed with `xN` notation. Section names from the parts track appear in `[brackets]`.

### Pattern detection

The algorithm scans left to right, greedily matching the smallest repeating unit at each position. For each position, it tries pattern lengths 1, 2, 3, ... and takes the first that repeats 2+ times consecutively. Unmatched bars are emitted as literals. This is O(n^2) worst case but fast enough for typical song lengths (< 200 bars).

In interleaved mode, patterns are detected on the primary track (the one with the most labels, typically chords). Labels from other tracks (e.g. section names from parts) split the repeat range and appear as `[annotations]` rather than breaking the pattern.

| Arg | Description |
|-----|-------------|
| `source` | Audio file (.mp3/.aup3) for auto-discovery |
| `-l` / `--labels` | Label files (overrides auto-discovery) |
| `--bars` | Bars grid file (overrides auto-discovery) |
| `--beats` | Beats grid file (overrides auto-discovery) |
| `-s` / `--separate` | Show each track separately |
| `-p` / `--show-prefix` | Show track prefix on each label |
| `-e` / `--expand` | Spell out every label |
| `--show-beats` | Always show beat number |

## Running tests

```bash
uv run --with pytest pytest -v
```

All tests are self-contained and run without external data.
