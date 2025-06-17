import argparse

from pydub import AudioSegment
import pydub.scipy_effects # Necesitado para high_pass_filter
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pathlib import Path

from plot_utils import (
    plot_audio,
    plot_audio_filtrado_con_picos,
    plot_fft
)

from utils import (
    frequency_spectrum,
    calculate_distance,
    classify_note_attempt_3,
)


def melody_recognition(file, note_file=None):
    # If a note file is supplied read them in
    actual_notes = []
    if note_file:
        with open(note_file) as f:
            for line in f:
                actual_notes.append(line.strip())

    song = AudioSegment.from_file(file)
    plot_audio(song, segment_ms=10, prefix='./melodia/01')
    starts = predict_note_starts(song)
    predicted_notes = predict_notes(song, starts, actual_notes)

    print("")
    if actual_notes:
        print("Actual Notes")
        print(actual_notes)
    print("Predicted Notes")
    print(predicted_notes)

    if actual_notes:
        lev_distance = calculate_distance(predicted_notes, actual_notes)
        print("Levenshtein distance: {}/{}".format(lev_distance, len(actual_notes)))


# Very simple implementation, just requires a minimum volume and looks for left edges by
# comparing with the prior sample, also requires a minimum distance between starts
# Future improvements could include smoothing and/or comparing multiple samples
#
# song: pydub.AudioSegment
#
# Returns perdicted starts in ms
def predict_note_starts(song):
    # Size of segments to break song into for volume calculations
    SEGMENT_MS = 10
    # Minimum volume necessary to be considered a note
    VOLUME_THRESHOLD = -35
    # The increase from one sample to the next required to be considered a note
    EDGE_THRESHOLD = 5
    # Throw out any additional notes found in this window
    MIN_MS_BETWEEN = 100

    # Filter out lower frequencies to reduce noise
    song = song.high_pass_filter(80, order=4)
    plot_audio(song, SEGMENT_MS, prefix='./melodia/02')
    # dBFS is decibels relative to the maximum possible loudness
    volume = [segment.dBFS for segment in song[::SEGMENT_MS]]

    predicted_starts = []
    for i in range(1, len(volume)):
        if volume[i] > VOLUME_THRESHOLD and volume[i] - volume[i - 1] > EDGE_THRESHOLD:
            ms = i * SEGMENT_MS
            # Ignore any too close together
            if len(predicted_starts) == 0 or ms - predicted_starts[-1] >= MIN_MS_BETWEEN:
                predicted_starts.append(ms)

    # Plot the volume over time (sec)
    plot_audio_filtrado_con_picos(volume, predicted_starts, SEGMENT_MS, prefix='./melodia/03')

    return predicted_starts


def predict_notes(song, starts, actual_notes):
    predicted_notes = []
    for i, start in enumerate(starts):
        sample_from = start + 50
        sample_to = start + 550
        if i < len(starts) - 1:
            sample_to = min(starts[i + 1], sample_to)
        segment = song[sample_from:sample_to]
        freqs, freq_magnitudes = frequency_spectrum(segment)
        plot_fft(freqs, freq_magnitudes, prefix='./melodia/04')

        predicted = classify_note_attempt_3(freqs, freq_magnitudes)
        predicted_notes.append(predicted or "U")

        # Print general info
        print("")
        print("Note: {}".format(i))
        if i < len(actual_notes):
            print("Predicted: {} Actual: {}".format(predicted, actual_notes[i]))
        else:
            print("Predicted: {}".format(predicted))
        print("Predicted start: {}".format(start))
        length = sample_to - sample_from
        print("Sampled from {} to {} ({} ms)".format(sample_from, sample_to, length))
        print("Frequency sample period: {}hz".format(freqs[1]))

        # Print peak info
        peak_indicies, props = scipy.signal.find_peaks(freq_magnitudes, height=0.015)
        print("Peaks of more than 1.5 percent of total frequency contribution:")
        for j, peak in enumerate(peak_indicies):
            freq = freqs[peak]
            magnitude = props["peak_heights"][j]
            print("{:.1f}hz with magnitude {:.3f}".format(freq, magnitude))

    return predicted_notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--note-file", type=str)
    args = parser.parse_args()

    Path("./melodia").mkdir(parents=True, exist_ok=True)

    melody_recognition(args.file, note_file=args.note_file)