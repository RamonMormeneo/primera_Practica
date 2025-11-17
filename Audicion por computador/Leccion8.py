#Classificadores de audio

# para quitar warnings 

from transformers import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore')

# importamos desde la librería transformers el pipeline

from transformers import pipeline

# definimos la tarea y el modelo

tarea = 'audio-classification'  

modelo = 'Yash03813/pingpong-music_genres_classification-finetuned-finetuned-gtzan'

# iniciamos el modelo, los modelos se descargan en local, en este caso son unos 400Mb

pipe = pipeline(task=tarea, model=modelo)

pipe('../../../files/jazz_sample.wav')

# definimos la tarea y el modelo

tarea = 'audio-classification'  

modelo = 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition'

# iniciamos el modelo, los modelos se descargan en local, en este caso son unos 1.3Gb

pipe = pipeline(task=tarea, model=modelo)

pipe('audio_cache/test_mixture_3spks.wav')

# definimos la tarea y el modelo

tarea = 'audio-classification'  

modelo = 'dima806/english_accents_classification'

# iniciamos el modelo, los modelos se descargan en local, en este caso son unos 400Mb

pipe = pipeline(task=tarea, model=modelo)

pipe('audio_cache/test_mixture_3spks.wav')