#Text to audio

# para quitar warnings 

from transformers import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore')

# importamos desde la librería transformers el pipeline

from transformers import pipeline

# definimos la tarea y el modelo

tarea = 'text-to-audio'  

modelo = 'facebook/musicgen-medium'

# iniciamos el modelo de generación de audio, los modelos se descargan en local, en este caso son unos 8Gb

audio_pipe = pipeline(task=tarea, model=modelo)

# descripoción de la música

prompt = 'jazz music with a piano melody, bass and drums.'

# parámetros para ajustar la salida


params = {'temperature': 0.7,      # cuanto más alto más "creatividad" (aleatorio)
          'max_new_tokens': 150    # cuanto más alto más larga la respuesta
         }

# llamada al pipeline con texto, la respuesta es un array de NumPy y la frecuencia de muestreo

respuesta = audio_pipe(text_inputs=prompt,
                       generate_kwargs=params)

respuesta

# escuchamos el audio

from IPython.display import Audio

Audio(data=respuesta['audio'][0][0], rate=respuesta['sampling_rate'])

# podemos guardar el audio en formato wav con scipy

import scipy

scipy.io.wavfile.write(filename='../../../files/jazz_sample.wav', 
                       rate=respuesta['sampling_rate'], 
                       data=respuesta['audio'][0][0])


#uso de modelo


# importamos los objetos procesador y modelo

from transformers import AutoProcessor, MusicgenForConditionalGeneration

# inicialización del procesador

procesador = AutoProcessor.from_pretrained(pretrained_model_name_or_path='facebook/musicgen-medium')

# procesamiento del texto, podemos pasar varios

tensor = procesador(text=['80s pop track with bassy drums and synth', 
                          '90s rock song with loud guitars and heavy drums'],
                    
                    padding=True,
                    return_tensors='pt',
                   )

tensor

# inicialización del modelo

modelo = MusicgenForConditionalGeneration.from_pretrained(pretrained_model_name_or_path='facebook/musicgen-medium')


# respuesta del modelo

respuesta = modelo.generate(**tensor, max_new_tokens=150)

respuesta

# escuchamos el primer audio

Audio(data=respuesta[0].detach().numpy(), rate=modelo.config.audio_encoder.sampling_rate)

# escuchamos el segundo audio

Audio(data=respuesta[1].detach().numpy(), rate=modelo.config.audio_encoder.sampling_rate)