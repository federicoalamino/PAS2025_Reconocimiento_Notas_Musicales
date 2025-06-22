import matplotlib.pyplot as plt
import numpy as np

def plot_audio(song, segment_ms=50, prefix='./acorde/01'):
    # Size of segments to break song into for volume calculations
    SEGMENT_MS = segment_ms
    # dBFS is decibels relative to the maximum possible loudness
    volume = [segment.dBFS for segment in song[::SEGMENT_MS]]
    x_axis = np.arange(len(volume)) * (SEGMENT_MS / 1000)
    plt.plot(x_axis, volume)
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Volumen")
    plt.savefig(f'{prefix}) audio.png')
    plt.close()

def plot_audio_filtrado_con_picos(volume, starts, segment_ms=10, prefix='./melodia/03'):
    x_axis = np.arange(len(volume)) * (segment_ms / 1000)
    plt.plot(x_axis, volume)

    # Se agrega cada pico como una linea vertical
    for s in starts:
        plt.axvline(x=(s / 1000), color="r", linewidth=0.5, linestyle="-")
    
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Volumen")

    plt.savefig(f'{prefix}) filtrado-con-picos.png')
    plt.close()
    

def plot_fft(freq_array, freq_magnitude, prefix='./acorde/02'):
    plt.plot(freq_array, freq_magnitude, 'b')
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Volumen")
    plt.savefig(f'{prefix}) fft.png')
    plt.close()

def plot_picos(freq_array, freq_magnitude, peaksIdx, prefix='./acorde/03'):
    plt.plot(freq_array, freq_magnitude, 'b')

    for s in freq_array[peaksIdx]:
        plt.axvline(x=s, color='r', linewidth=0.5, linestyle="-")
    
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Volumen")

    plt.savefig(f'{prefix}) fft-con-picos.png')
    plt.close()

def plot_nota_peso(lista_nota_peso, prefix='./acorde/04'):
    notas = list(set([x[0] for x in lista_nota_peso]))
    y = []

    for nota in notas:
        peso = 0
        for nota_peso in lista_nota_peso:
            if nota == nota_peso[0]:
                peso += nota_peso[1]
        y.append(peso)

    for i in range(len(notas)):
        if notas[i] == None:
            notas[i] = "No reconocido"

    plt.pie(y, labels=notas, autopct='%.0f%%')
    plt.savefig(f'{prefix}) nota-peso.png')
    plt.close()