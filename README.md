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
# DBNDownBeatTracker gives beats WITH bar positions (1, 2, 3, 4)
DBNDownBeatTracker single new_audio.mp3 > new_downbeats.txt

# Extract beat grid (all beats)
awk '{printf "%.6f\n", $1}' new_downbeats.txt > new_beats.txt

# Extract bar grid (downbeats only, as Audacity point labels)
awk '$2==1 {n++; printf "%.6f\t%.6f\t%d\n", $1, $1, n}' new_downbeats.txt > new_bars.txt
```

### 2. Run remap_labels

```bash
remap_labels.py \
    old_audio.mp3 \
    new_audio.mp3 \
    -b new_beats.txt \
    -B new_bars.txt \
    --old-beats beats_old.txt \
    --old-bars bars_old.txt \
    chords_old.txt parts_old.txt guit_old.txt rehearse_old.txt
```

(Assumes `remap_labels.py` is on your PATH - see Installation.)

All label files listed at the end are remapped. Output goes to `remapped/` (override with `-o`).

#### Arguments

| Arg | Description |
|-----|-------------|
| `old_audio` | Original audio file |
| `new_audio` | New (replacement) audio file |
| `-b` / `--new-beats` | New beat grid (from DBNDownBeatTracker) |
| `-B` / `--new-bars` | New bar grid (downbeats from DBNDownBeatTracker) |
| `--old-beats` | Old beat grid |
| `--old-bars` | Old bar grid |
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
DBNDownBeatTracker single 21_Everybody_needs_somebody_Playback_band.mp3 > new_downbeats.txt
awk '{printf "%.6f\n", $1}' new_downbeats.txt > new_beats.txt
awk '$2==1 {n++; printf "%.6f\t%.6f\t%d\n", $1, $1, n}' new_downbeats.txt > new_bars.txt

# Step 2: remap
remap_labels.py \
    blues_brothers_everybody_needs_somebody.mp3 \
    21_Everybody_needs_somebody_Playback_band.mp3 \
    -b new_beats.txt \
    -B new_bars.txt \
    --old-beats beats_blues_brothers_everybody_needs_somebody.txt \
    --old-bars bars_blues_brothers_everybody_needs_somebody.txt \
    chords_blues_brothers_everybody_needs_somebody.txt \
    parts_blues_brothers_everybody_needs_somebody.txt \
    guit_blues_brothers_everybody_needs_somebody.txt

# Step 3: import remapped/*.txt into Audacity with the new audio
```

## Running tests

```bash
uv run --with pytest pytest test_remap_labels.py -v
```

Some tests require the real song data to be present and a prior run of `remap_labels.py` to have generated output in `remapped/`. These are skipped automatically if the data isn't available.
