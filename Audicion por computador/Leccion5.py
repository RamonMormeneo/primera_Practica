#Text to speech
# para quitar warnings 

from transformers import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore')

# importamos desde la librería transformers el pipeline

from transformers import pipeline

# definimos la tarea y el modelo

tarea = 'text-to-speech'  

modelo = 'facebook/mms-tts-spa'

# iniciamos el modelo TTS, los modelos se descargan en local, en este caso son unos 150Mb

tts_pipe = pipeline(task=tarea, model=modelo)

# llamada al pipeline con texto, la respuesta es un array de NumPy y la frecuencia de muestreo

respuesta = tts_pipe('Hola')

# escuchamos el audio

from IPython.display import Audio

Audio(respuesta['audio'], rate=respuesta['sampling_rate'])

#uso de modelo

# importamos los objetos tokenizador y modelo

from transformers import AutoTokenizer, VitsModel

tokenizador = AutoTokenizer.from_pretrained('facebook/mms-tts-spa')

# texto a tokenizar

texto = 'hola que tal como estas'

# tokenizar texto

tensor = tokenizador(text=texto, return_tensors='pt')

#uso de VitsModel

modelo = VitsModel.from_pretrained('facebook/mms-tts-spa')

respuesta = modelo(**tensor)

audio = respuesta.waveform

Audio(audio.detach().numpy(), rate=modelo.config.sampling_rate)

#guardar en wav

import scipy

scipy.io.wavfile.write(filename='../../files/tts_audio.wav', rate=modelo.config.sampling_rate, data=audio.detach().numpy().T)

