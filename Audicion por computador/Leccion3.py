#Librosa

# para quitar warnings 

import warnings
warnings.filterwarnings('ignore')

# importamos la librería

import librosa

# ruta a la muestra de audio

ruta = librosa.example('brahms')

# cargamos la muestra

data, freq = librosa.load(ruta)

# dimensiones del array

data.shape

# frecuencia de muestreo en Hz

freq

# escuchar el audio

from IPython.display import Audio

Audio(data=data, rate=freq)

#visualizar forma de ola

# importamos pylab

import pylab as plt

# se inicia la figura con dimensiones 140x50 píxeles
plt.figure(figsize=(14, 5))

# se carga con librosa la forma del audio
librosa.display.waveshow(y=data, sr=freq)

# título del gráfico
plt.title('Forma de Onda del Audio')

# etiqueta eje x
plt.xlabel('Tiempo')

# etiqueta eje y
plt.ylabel('Amplitud')

# mostramos la figura
plt.show();

#Recorte y ganancia

# recorte del audio
data = data[13000:100000]


# se inicia la figura con dimensiones 140x50 píxeles
plt.figure(figsize=(14, 5))

# se carga con librosa la forma del audio
librosa.display.waveshow(y=data, sr=freq)

# título del gráfico
plt.title('Forma de Onda del Audio')

# etiqueta eje x
plt.xlabel('Tiempo')

# etiqueta eje y
plt.ylabel('Amplitud')

# mostramos la figura
plt.show();

# factor de ganancia
ganancia = 1.5

# aumentando el volumen del audio
data = ganancia * data

# se inicia la figura con dimensiones 140x50 píxeles
plt.figure(figsize=(14, 5))

# se carga con librosa la forma del audio
librosa.display.waveshow(y=data, sr=freq)

# título del gráfico
plt.title('Forma de Onda del Audio')

# etiqueta eje x
plt.xlabel('Tiempo')

# etiqueta eje y
plt.ylabel('Amplitud')

# mostramos la figura
plt.show();


# normalizando el audio
data = data / max(abs(data))

# se inicia la figura con dimensiones 140x50 píxeles
plt.figure(figsize=(14, 5))

# se carga con librosa la forma del audio
librosa.display.waveshow(y=data, sr=freq)

# título del gráfico
plt.title('Forma de Onda del Audio')

# etiqueta eje x
plt.xlabel('Tiempo')

# etiqueta eje y
plt.ylabel('Amplitud')

# mostramos la figura
plt.show();

#Transformaciones basicas

# importamos numpy

import numpy as np

# calculo de la STFT

stft = librosa.stft(y=data)

stft.shape

# convierteun espectrograma en amplitud a uno en escala decibélica 

db = librosa.amplitude_to_db(S=np.abs(stft), ref=np.max)

db.shape

# se inicia la figura con dimensiones 140x50 píxeles
plt.figure(figsize=(14, 5))

# se carga con librosa  el espectrograma
librosa.display.specshow(data=db, sr=freq, x_axis='time', y_axis='log')

# barra de color lateral
plt.colorbar(format='%+2.0f dB')


# título del gráfico
plt.title('Espectrograma STFT')

# etiqueta eje x
plt.xlabel('Tiempo - seg')

# etiqueta eje y
plt.ylabel('Frecuencia - Hz')

# mostramos la figura
plt.show();