import array
from collections import Counter

import numpy as np
from scipy.fft import fft
from pydub.utils import get_array_type
from Levenshtein import distance
from itertools import combinations

NOTAS = {
    "A": 440,
    "A#": 466.1637615180899,
    "B": 493.8833012561241,
    "C": 523.2511306011972,
    "C#": 554.3652619537442,
    "D": 587.3295358348151,
    "D#": 622.2539674441618,
    "E": 659.2551138257398,
    "F": 698.4564628660078,
    "F#": 739.9888454232688,
    "G": 783.9908719634985,
    "G#": 830.6093951598903,
}

# Devuelve una lista de frecuencias y una lista de cuan frecuente esa frecuencia esta en la muestra
def frequency_spectrum(sample, max_frequency=800):
    # Se convierte de pydub.AudioSample a un raw audio data
    # https://stackoverflow.com/questions/32373996/pydub-raw-audio-data
    bit_depth = sample.sample_width * 8
    array_type = get_array_type(bit_depth)
    raw_audio_data = array.array(array_type, sample._data)
    n = len(raw_audio_data)

    # Calcula FFT y frevuencia para cada indice en la lista de FFT
    # https://stackoverflow.com/questions/53308674/audio-frequencies-in-python
    freq_array = np.arange(n) * (float(sample.frame_rate) / n)
    freq_array = freq_array[: (n // 2)]

    raw_audio_data = raw_audio_data - np.average(raw_audio_data)
    freq_magnitude = fft(raw_audio_data)
    freq_magnitude = freq_magnitude[: (n // 2)]

    if max_frequency:
        max_index = int(max_frequency * n / sample.frame_rate) + 1
        freq_array = freq_array[:max_index]
        freq_magnitude = freq_magnitude[:max_index]

    freq_magnitude = abs(freq_magnitude)
    freq_magnitude = freq_magnitude / np.sum(freq_magnitude)
    return freq_array, freq_magnitude


def clasificador_de_nota(freq_array, freq_magnitude):
    min_freq = 82
    note_counter = Counter()
    for i in range(len(freq_magnitude)):
        if freq_magnitude[i] < 0.01:
            continue

        for freq_multiplier, credit_multiplier in [
            (1, 1),
            (1 / 3, 3 / 4),
            (1 / 5, 1 / 2),
            (1 / 6, 1 / 2),
            (1 / 7, 1 / 2),
        ]:
            freq = freq_array[i] * freq_multiplier
            if freq < min_freq:
                continue
            note = obtener_la_nota_de_la_frecuencia(freq)
            if note:
                note_counter[note] += freq_magnitude[i] * credit_multiplier

    return note_counter.most_common(1)[0][0]


# si la frecuencia esta dentro de la tolerancia de la nota (medida en cents - 1/100th de un semitono)
# retorna esa nota, de lo contrario retorna None
# Se escala todo a la octava de 440hz para chequear
def obtener_la_nota_de_la_frecuencia(f, tolerance=33):
    # Se calcula el rango para cada nota
    tolerance_multiplier = 2 ** (tolerance / 1200)
    note_ranges = {
        k: (v / tolerance_multiplier, v * tolerance_multiplier) for (k, v) in NOTAS.items()
    }

    # Se obtiene la frecuencia dentro de 440hz
    range_min = note_ranges["A"][0]
    range_max = note_ranges["G#"][1]
    if f < range_min:
        while f < range_min:
            f *= 2
    else:
        while f > range_max:
            f /= 2

    # Retorna la nota si se encuentra dentro del rango
    for (note, note_range) in note_ranges.items():
        if f > note_range[0] and f < note_range[1]:
            return note
    return None


# Asume que no hay bemoles.
def calcular_distancia_levenshtein(predicted, actual):
    # Para simplificar, las notas naturales (sin el sostenido) estan en lower case
    # y las sostenidas en upper case.
    def transform(note):
        if "#" in note:
            return note[0].upper()
        return note.lower()

    return distance(
        "".join([transform(n) for n in predicted]), "".join([transform(n) for n in actual]),
    )
