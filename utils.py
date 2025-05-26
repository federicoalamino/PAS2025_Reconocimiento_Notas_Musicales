import array
from collections import Counter

import numpy as np
from scipy.fft import fft
from pydub.utils import get_array_type
from Levenshtein import distance
from itertools import combinations

NOTES = {
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

NOTAS_PESO = {
    "A": 0,
    "A#": 0,
    "B": 0,
    "C": 0,
    "C#": 0,
    "D": 0,
    "D#": 0,
    "E": 0,
    "F": 0,
    "F#": 0,
    "G": 0,
    "G#": 0,
}

CHORDS = {
    "CMajor":   ["C",  "E",  "G"],
    "C#Major":  ["C#", "F",  "G#"],
    "DMajor":   ["D",  "F#", "A"],
    "D#Major":  ["D#", "G",  "A#"],
    "EMajor":   ["E",  "G#", "B"],
    "FMajor":   ["F",  "A",  "C"],
    "F#Major":  ["F#", "A#", "C#"],
    "GMajor":   ["G",  "B",  "D"],
    "G#Major":  ["G#", "C",  "D#"],
    "AMajor":   ["A",  "C#", "E"],
    "A#Major":  ["A#", "D",  "F"],
    "BMajor":   ["B",  "D#", "F#"],
    "Cminor":   ["C",  "D#", "G"],
    "C#minor":  ["C#", "E",  "G#"],
    "Dminor":   ["D",  "F",  "A"],
    "D#minor":  ["D#", "F#", "A#"],
    "Eminor":   ["E",  "G",  "B"],
    "Fminor":   ["F",  "G#", "C"],
    "F#minor":  ["F#", "A",  "C#"],
    "Gminor":   ["G",  "A#", "D"],
    "G#minor":  ["G#", "B",  "D#"],
    "Aminor":   ["A",  "C",  "E"],
    "A#minor":  ["A#", "C#", "F"],
    "Bminor":   ["B",  "D",  "F#"]
}


def frequency_spectrum(sample, max_frequency=800):
    """
    Derive frequency spectrum of a signal pydub.AudioSample
    Returns an array of frequencies and an array of how prevelant that frequency is in the sample
    """
    # Convert pydub.AudioSample to raw audio data
    # Copied from Jiaaro's answer on https://stackoverflow.com/questions/32373996/pydub-raw-audio-data
    bit_depth = sample.sample_width * 8
    array_type = get_array_type(bit_depth)
    raw_audio_data = array.array(array_type, sample._data)
    n = len(raw_audio_data)

    # Compute FFT and frequency value for each index in FFT array
    # Inspired by Reveille's answer on https://stackoverflow.com/questions/53308674/audio-frequencies-in-python
    freq_array = np.arange(n) * (float(sample.frame_rate) / n)  # two sides frequency range
    freq_array = freq_array[: (n // 2)]  # one side frequency range

    raw_audio_data = raw_audio_data - np.average(raw_audio_data)  # zero-centering
    freq_magnitude = fft(raw_audio_data)  # fft computing and normalization
    freq_magnitude = freq_magnitude[: (n // 2)]  # one side

    if max_frequency:
        max_index = int(max_frequency * n / sample.frame_rate) + 1
        freq_array = freq_array[:max_index]
        freq_magnitude = freq_magnitude[:max_index]

    freq_magnitude = abs(freq_magnitude)
    freq_magnitude = freq_magnitude / np.sum(freq_magnitude)
    return freq_array, freq_magnitude


def classify_note_attempt_1(freq_array, freq_magnitude):
    i = np.argmax(freq_magnitude)
    f = freq_array[i]
    print("frequency {}".format(f))
    print("magnitude {}".format(freq_magnitude[i]))
    return get_note_for_freq(f)


def classify_note_attempt_2(freq_array, freq_magnitude):
    note_counter = Counter()
    for i in range(len(freq_magnitude)):
        if freq_magnitude[i] < 0.01:
            continue
        note = get_note_for_freq(freq_array[i])
        if note:
            note_counter[note] += freq_magnitude[i]
    return note_counter.most_common(1)[0][0]


def classify_note_attempt_3(freq_array, freq_magnitude):
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
            note = get_note_for_freq(freq)
            if note:
                note_counter[note] += freq_magnitude[i] * credit_multiplier

    return note_counter.most_common(1)[0][0]


# If f is within tolerance of a note (measured in cents - 1/100th of a semitone)
# return that note, otherwise returns None
# We scale to the 440 octave to check
def get_note_for_freq(f, tolerance=33):
    # Calculate the range for each note
    tolerance_multiplier = 2 ** (tolerance / 1200)
    note_ranges = {
        k: (v / tolerance_multiplier, v * tolerance_multiplier) for (k, v) in NOTES.items()
    }

    # Get the frequence into the 440 octave
    range_min = note_ranges["A"][0]
    range_max = note_ranges["G#"][1]
    if f < range_min:
        while f < range_min:
            f *= 2
    else:
        while f > range_max:
            f /= 2

    # Check if any notes match
    for (note, note_range) in note_ranges.items():
        if f > note_range[0] and f < note_range[1]:
            return note
    return None


# Assumes everything is either natural or sharp, no flats
# Returns the Levenshtein distance between the actual notes and the predicted notes
def calculate_distance(predicted, actual):
    # To make a simple string for distance calculations we make natural notes lower case
    # and sharp notes cap
    def transform(note):
        if "#" in note:
            return note[0].upper()
        return note.lower()

    return distance(
        "".join([transform(n) for n in predicted]), "".join([transform(n) for n in actual]),
    )


def obtener_lista_nota_peso_usando(freq_array, freq_magnitudes, peaksIdx):
    tuplas_nota_magnitud = []
    for x,y in zip(freq_array[peaksIdx], freq_magnitudes[peaksIdx]):
        tuplas_nota_magnitud.append((get_note_for_freq(x),y))

    tuplas_nota_magnitud_ordenado_por_magnitud = sorted(tuplas_nota_magnitud, key=lambda x: x[1])
    tuplas_nota_peso_ordenada = [ (tuplas_nota_magnitud_ordenado_por_magnitud[x][0], x) for x in range(len(tuplas_nota_magnitud_ordenado_por_magnitud))]

    # print(tuplas_nota_peso_ordenada)

    # Devolvemos la mitad con peso mayor para sacarnos las frecuencias mas bajas
    # return [x for x in tuplas_nota_peso_ordenada[(len(tuplas_nota_peso_ordenada)//2):]]
    return tuplas_nota_peso_ordenada

def agregar_nota_y_peso(nota_y_peso):
    nota = nota_y_peso[0]
    peso = nota_y_peso[1]
    
    if nota == None:
        return
    
    NOTAS_PESO[nota] = NOTAS_PESO[nota] + peso

def imprimir_notas_peso():
    print(NOTAS_PESO)

def obtener_combinaciones():
    notas_usadas = []
    for nota, peso in NOTAS_PESO.items():
        if peso > 0:
            notas_usadas.append(nota)
    
    return list(combinations(notas_usadas, 3))

def posibles_acordes(combinaciones):
    acordes = {}
    for combinacion in combinaciones:
        acorde = acorde_de(combinacion)
        if acorde != None:
            acordes[acorde] = (combinacion, peso_total_de(combinacion))
    return acordes

def acorde_de(combinacion):
    combinacion_ordenada = sorted(combinacion)
    for acorde, notas in CHORDS.items():
        if sorted(notas) == combinacion_ordenada:
            return acorde
    return None

def peso_total_de(combinacion):
    total = 0
    for nota in combinacion:
        total += NOTAS_PESO[nota]
    return total

def mostrar_posibles_ordenados(dicAcordes):
    maxProb = 0
    acorde_max_prob = {}
    for key,val in dicAcordes.items():
        if val[1] > maxProb:
            maxProb = val[1]
            acorde_max_prob = (key, val[0])
    
    hastaProb = maxProb // 2
    otros_posibles = []
    for key,val in dicAcordes.items():
        if val[1] > hastaProb and val[1] != maxProb:
            otros_posibles.append((key, val))
    otros_posibles = sorted(otros_posibles, key=lambda x: x[1][1], reverse=True)
    
    imprimir_en_verde(f"Acorde Predicho: {acorde_max_prob[0]} ({acorde_max_prob[1][0]} {acorde_max_prob[1][1]} {acorde_max_prob[1][2]})")
    
    if len(otros_posibles) > 0:
        imprimir_en_amarillo("Otros posibles en orden de probabilidad")
        for acorde in otros_posibles:
            imprimir_en_amarillo(f"{acorde[0]} ({acorde[1][0][0]} {acorde[1][0][1]} {acorde[1][0][2]}) {acorde[1][1]}")
    else:
        imprimir_en_amarillo("No se predijeron mas acordes")

def imprimir_en_verde(texto):
    print('\033[102m' + texto + '\033[0m')

def imprimir_en_amarillo(texto):
    print('\033[93m' + texto + '\033[0m')
