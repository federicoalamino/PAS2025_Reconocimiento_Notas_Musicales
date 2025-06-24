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
    clasificador_de_nota,
    calcular_distancia_levenshtein
)


def predecir_notas_de_melodia(file, note_file=None):
    # Se levanta el archivo de notas si existe
    actual_notes = []
    if note_file:
        with open(note_file) as f:
            for line in f:
                actual_notes.append(line.strip())

    song = AudioSegment.from_file(file)
    plot_audio(song, segment_ms=10, prefix='./melodia/01')
    starts = predecir_comienzos_de_notas(song)
    predicted_notes = predecir_notas(song, starts, actual_notes)

    print("")
    if actual_notes:
        print("Notas Reales")
        print(actual_notes)
    print("Notas Predichas")
    print(predicted_notes)

    if actual_notes:
        lev_distance = calcular_distancia_levenshtein(predicted_notes, actual_notes)
        print("Distancia de Levenshtein: {}/{}".format(lev_distance, len(actual_notes)))


# Los comienzos predichos estan en ms
def predecir_comienzos_de_notas(song):
    # Tamaño de los chunks tomados en la cancion para calcular los volumenes
    SEGMENT_MS = 10
    # El minimo volumen necesario para que se considere que es una nota
    VOLUME_THRESHOLD = -35
    # El tamaño de una muestra a la otra para ser considerada una nota
    EDGE_THRESHOLD = 5
    # Descarta notas adicionales que se encuentran en esta ventana
    MIN_MS_BETWEEN = 100

    # Se filtran las frecuencias bajas para reducir el ruido
    song = song.high_pass_filter(80, order=4)
    plot_audio(song, SEGMENT_MS, prefix='./melodia/02')
    # dBFS: decibelios a escala completa, relativo al maximo posible de volumen
    volume = [segment.dBFS for segment in song[::SEGMENT_MS]]

    predicted_starts = []
    for i in range(1, len(volume)):
        if volume[i] > VOLUME_THRESHOLD and volume[i] - volume[i - 1] > EDGE_THRESHOLD:
            ms = i * SEGMENT_MS
            # Se ignoran los que estan demasiado cerca
            if len(predicted_starts) == 0 or ms - predicted_starts[-1] >= MIN_MS_BETWEEN:
                predicted_starts.append(ms)

    plot_audio_filtrado_con_picos(volume, predicted_starts, SEGMENT_MS, prefix='./melodia/03')

    return predicted_starts


def predecir_notas(song, starts, actual_notes):
    predicted_notes = []
    for i, start in enumerate(starts):
        sample_from = start + 50
        sample_to = start + 550
        if i < len(starts) - 1:
            sample_to = min(starts[i + 1], sample_to)
        segment = song[sample_from:sample_to]
        freqs, freq_magnitudes = frequency_spectrum(segment)
        plot_fft(freqs, freq_magnitudes, prefix='./melodia/04')

        predicted = clasificador_de_nota(freqs, freq_magnitudes)
        predicted_notes.append(predicted or "U")

        # Print general info
        print("")
        print("Nota: {}".format(i))
        if i < len(actual_notes):
            print("Predicha: {} Real: {}".format(predicted, actual_notes[i]))
        else:
            print("Predicha: {}".format(predicted))
        print("Comienzo predicho: {}".format(start))
        length = sample_to - sample_from
        print("Muestra de {} hasta {} ({} ms)".format(sample_from, sample_to, length))
        print("Periodo de frecuencia de muestra: {}hz".format(freqs[1]))

        # Print peak info
        peak_indicies, props = scipy.signal.find_peaks(freq_magnitudes, height=0.015)
        print("Picos de mas de 1.5 porciento del total de la frecuencia:")
        for j, peak in enumerate(peak_indicies):
            freq = freqs[peak]
            magnitude = props["peak_heights"][j]
            print("{:.1f}hz con magnitud {:.3f}".format(freq, magnitude))

    return predicted_notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--note-file", type=str)
    args = parser.parse_args()

    Path("./melodia").mkdir(parents=True, exist_ok=True)

    predecir_notas_de_melodia(args.file, note_file=args.note_file)