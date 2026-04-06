# Chroma Features and Dynamic Time Warping in remap_labels

## The problem

You have two recordings of the same song - an old version and a new version. The new version might be slightly faster, slower, or have small structural changes. You need to figure out: "this moment in the old recording corresponds to *that* moment in the new recording."

This is an audio alignment problem. You can't just compare waveforms directly because even tiny tempo changes make the waveforms completely different. You need a representation that captures *what's being played* rather than the exact shape of the sound wave.

## Chroma features: what is the music doing?

A chroma feature (also called a chromagram) reduces audio to its pitch content. At each moment in time, it answers: "how much energy is in each of the 12 pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)?"

It works like this:

1. Slice the audio into short overlapping windows (in our case, every 10ms).
2. For each window, compute a frequency spectrum (which frequencies are present).
3. Fold all octaves together - a C2, C3, C4 all count as "C".
4. You get a 12-dimensional vector per window: one value per pitch class.

The result is a matrix: 12 rows (pitch classes) by N columns (time windows). When a C major chord is playing, the C, E, and G rows light up. When it changes to F major, F, A, and C light up instead.

Chroma features are robust to:
- Timbre differences (same chord on piano vs guitar looks similar)
- Octave differences (bass playing C2 and vocal singing C4 both register as "C")
- Volume changes

They're sensitive to:
- Pitch content (which is exactly what we want)
- Key changes (transposing a song shifts the entire chromagram)

In remap_labels, we use `librosa.feature.chroma_cqt` which computes chroma from a Constant-Q Transform - a frequency analysis where frequency bins are spaced logarithmically (matching musical pitch spacing), giving cleaner pitch detection than a standard FFT.

## Dynamic Time Warping: aligning two sequences

Given two chromagrams (old and new), Dynamic Time Warping finds the best alignment between them. It answers: "which frame in the old recording best matches which frame in the new recording?"

The key insight: the two sequences might not be the same length, and the mapping doesn't have to be uniform. One section might be stretched (slower tempo) while another is compressed (faster tempo). DTW handles this naturally.

### How it works

Imagine a grid where the X axis is time in the old recording and the Y axis is time in the new recording. Each cell (i, j) represents "how well does old frame i match new frame j?" - measured by the distance between their chroma vectors (we use cosine distance).

DTW finds a path through this grid from bottom-left to top-right that:
- Minimizes the total distance (follows the best-matching frames)
- Moves only forward in time (no going backwards)
- Can move diagonally (both advance), horizontally (old advances, new stays), or vertically (new advances, old stays)

The result is a warping path: a list of pairs (old_frame, new_frame) that maps every moment in one recording to its best match in the other.

### From path to time mapping

The warping path gives frame indices. We convert those to timestamps using the hop size (10ms per frame). Then we build an interpolation function: given any timestamp in the old recording, interpolate along the path to get the corresponding timestamp in the new recording.

This is the `warp` function in the code. `warp(45.3)` might return `46.1`, meaning "what was at 45.3 seconds in the old recording is now at 46.1 seconds in the new recording."

## How remap_labels uses DTW

In v6, DTW serves two limited purposes:

### 1. Positioning the first section

The tool needs to know where the song starts in the new recording. It warps the old first-section start time to get the new first-section start time, then snaps to the nearest bar boundary. After that, all subsequent sections are placed contiguously using their bar counts from the old data - no DTW involved.

Why not use DTW for every section? Because DTW has roughly 1-beat accuracy over long stretches. That's fine for finding "verse1 starts around here" (section-level), but not precise enough for "this chord starts at beat 3 of bar 19" (beat-level). The musical structure handles beat-level precision; DTW handles the coarse positioning.

### 2. Detecting structural anomalies

DTW scans the warping path for regions where time compresses or expands abnormally:

- **Compression**: a 4-second stretch of old audio maps to less than 1.2 seconds of new audio. This suggests a section was removed in the new version.
- **Expansion**: a 4-second stretch of new audio maps to less than 1.2 seconds of old audio. This suggests a section was added in the new version.

These anomalies are reported in the review label track so you know where to check manually.

## Computational cost

DTW's time complexity is O(N * M) where N and M are the number of frames in each recording. For a 200-second song at 10ms hop, that's ~20,000 frames per recording, so ~400 million cells. librosa's implementation uses optimized C code, but it's still the slowest part of the pipeline (a few seconds per song). The musical structure reconstruction itself is nearly instant.

## Limitations

- **Same key**: chroma features assume both recordings are in the same key. A transposed version would look different and DTW would struggle.
- **Pitched content only**: sections with only drums or speech have weak chroma features, making DTW less reliable there. This is why the "intro_speech" section in Everybody Needs Somebody sometimes aligns imprecisely.
- **Not beat-aware**: DTW aligns audio content, not musical structure. It doesn't know about bars, beats, or chords. That's why remap_labels uses it only for coarse positioning and relies on the bar/beat grids for precise placement.
