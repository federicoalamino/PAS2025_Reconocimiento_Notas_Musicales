# TP - Procesamiento avanzado de señales

## Reconocimiento de notas de una melodía

### Entrada

```bash
python3 predecir_notas_de_melodias.py "archivo de audio" [*optional* --note-file "archivo de texto con notas"]
```

Solo pasandole el nombre de un archivo musical funciona. Ejemplo:
``` bash
python3 predecir_notas_de_melodias.py arroz-con-leche.mp3
```
Esto devuelve las predicciones de cada nota. Si le pasamos un archivo de las notas te compara que nota predijo con la nota actual.
Ejemplo:
``` bash
python3 predecir_notas_de_melodias.py arroz-con-leche.mp3 --note-file arroz-con-leche-notas
```
Este archivo de notas es de la forma
```
A
C
...
```
Donde en cada linea tiene el nombre de una nota en notación americana, y en mayusculas.

### Salida
Plotea gráficos dentro de la carpeta `/acorde` (si no existe, la crea). Se encuentra el del audio,
el audio con los picos y el fft de uno de los picos.

Imprime por consola la predicción de cada nota. Si se le pasa un archivo con las notas reales,
muestra ambas listas y la distancia de Levenshtein entre ellas.

## Reconocimiento de acordes

### Entrada

Se le pasa el nombre de un archivo musical.
``` bash
python3 reconocer_acorde.py "archivo de audio"
```

Ejemplo:
``` bash
python3 reconocer_acorde.py CMajor.mp3
```

### Salida

Imprime por consola el acorde predicho con su peso. Además, imprime los demas acordes que reconoce
con sus pesos.

Plotea gráficos en la carpeta `/melodia` (si no existe la crea). Plotea el audio, el fft del audio,
el fft con los picos, y un grafico de torta con las notas y el porcentaje que aparecen en el audio.