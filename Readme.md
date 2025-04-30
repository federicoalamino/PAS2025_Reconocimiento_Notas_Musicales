# TP Procesamiento de señales

Como funciona:

```bash
python3 note_recognition.py "filename" [*optional* --note-file "a note file"] [*optional* --note-starts-file "a note starts file"] [*optional* --plot-starts "boolean", default=True] [*optional* --plot-fft-index "int"]
```

Solo pasandole el nombre de un archivo musical funciona. Ejemplo:
``` bash
python3 note_recognition.py redemption_song.m4a
```
Esto devuelve las predicciones de cada nota. Si le pasamos un archivo de las notas te compara que nota predijo con la nota actual.
Ejemplo:
``` bash
python3 note_recognition.py redemption_song.m4a --note-file redemption_song_notes
```
Este archivo de notas es de la forma
```
A
C
...
```
Donde en cada linea tiene el nombre de una nota en notación americana, y en mayusculas.