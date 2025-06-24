import argparse

from pydub import AudioSegment
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path

from plot_utils import (
    plot_audio,
    plot_fft,
    plot_picos,
    plot_nota_peso
)

from utils import (
    frequency_spectrum,
    obtener_la_nota_de_la_frecuencia
)

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

def obtener_lista_nota_peso_usando(freq_array, freq_magnitudes, peaksIdx):
    tuplas_nota_magnitud = []
    for x,y in zip(freq_array[peaksIdx], freq_magnitudes[peaksIdx]):
        tuplas_nota_magnitud.append((obtener_la_nota_de_la_frecuencia(x),y))

    tuplas_nota_magnitud_ordenado_por_magnitud = sorted(tuplas_nota_magnitud, key=lambda x: x[1])
    tuplas_nota_peso_ordenada = [ (tuplas_nota_magnitud_ordenado_por_magnitud[x][0], x) for x in range(len(tuplas_nota_magnitud_ordenado_por_magnitud))]

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
    
    print("\n")
    imprimir_en_verde(f"Acorde Predicho: {acorde_max_prob[0]} ({acorde_max_prob[1][0]} {acorde_max_prob[1][1]} {acorde_max_prob[1][2]}); peso: {maxProb}")
    print("\n")

    if len(otros_posibles) > 0:
        imprimir_en_amarillo("Otros posibles en orden de probabilidad")
        for acorde in otros_posibles:
            imprimir_en_amarillo(f"{acorde[0]} ({acorde[1][0][0]} {acorde[1][0][1]} {acorde[1][0][2]}); peso: {acorde[1][1]}")
    else:
        imprimir_en_amarillo("No se predijeron mas acordes")


def imprimir_en_verde(texto):
    print('\33[102m' + texto + '\33[0m')

def imprimir_en_amarillo(texto):
    print('\33[93m' + texto + '\33[0m')

def reconocer_acorde(file):
    song = AudioSegment.from_file(file)
    plot_audio(song)

    freq_array, freq_magnitudes = frequency_spectrum(song)
    plot_fft(freq_array, freq_magnitudes)

    peaksIdx, _ = find_peaks(freq_magnitudes, distance=150)
    plot_picos(freq_array, freq_magnitudes, peaksIdx)

    lista_nota_peso = obtener_lista_nota_peso_usando(freq_array, freq_magnitudes, peaksIdx)
    plot_nota_peso(lista_nota_peso)

    for nota_peso in lista_nota_peso:
        agregar_nota_y_peso(nota_peso)

    combinaciones = obtener_combinaciones()
    mostrar_posibles_ordenados(posibles_acordes(combinaciones))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()

    Path("./acorde").mkdir(parents=True, exist_ok=True)

    reconocer_acorde(args.file)