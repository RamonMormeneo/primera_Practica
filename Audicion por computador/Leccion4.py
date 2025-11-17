#Speech to Text

# para quitar warnings 

from transformers import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore')

# importamos desde la librería transformers el pipeline

from transformers import pipeline

# definimos la tarea y el modelo

tarea = 'automatic-speech-recognition'  

modelo = 'openai/whisper-base'

# iniciamos el modelo STT, los modelos se descargan en local, en este caso son unos 3.1Gb large-v3, 300Mb base

stt_pipe = pipeline(task=tarea, 
                    model=modelo, 
                    device='cpu')

# importamos la carga de datasets

from datasets import load_dataset

# dataset de prueba para whisper

dataset = load_dataset(path='distil-whisper/librispeech_long')

# tipo de datos del audio

audio = dataset['validation'][0]['audio']['array']

type(audio)

# lectura del audio

from IPython.display import Audio

Audio(data=audio[:], rate=16000)

resultado = stt_pipe(inputs=audio)

# importamos pytube

from pytube import YouTube

# Will AI kill everyone? Here's what the godfathers of AI have to say - 3:54

video_url = 'https://www.youtube.com/watch?v=NqmUBZQhOYw'

# usamos pytube para extraer el video

youtube = YouTube(video_url)

# ahora extraemos el audio desde el video

audio_youtube = youtube.streams.filter(only_audio=True).first()

audio_youtube

# descargamos el audio de YouTube...

ruta = '../../../files'

audio_youtube.download(ruta)


archivo = '/Will AI kill everyone Heres what the godfathers of AI have to say.mp4'


resultado = stt_pipe(inputs=ruta + archivo)

#Timestamp


resultado = stt_pipe(inputs=ruta + archivo, return_timestamps=True)

# texto asociado al timestamp

resultado['chunks'][:5]

#Generar subtitulos

# importamos timedelta para manejo temporal 
from datetime import timedelta


# frases con sus respectivos timestamps
chunks = resultado['chunks']


# bucle para cada frase, solo 4 como ejemplo
for i,e in enumerate(chunks[:4]):
    
    # inicio y final temporal del texto
    inicio = str(timedelta(seconds=int(e['timestamp'][0])))+',000'
    final = str(timedelta(seconds=int(e['timestamp'][1])))+',000'
    
    # texto extraido
    texto = e['text']
    
    # formato del subtítulo
    subtitulo = f'{i+1}\n{inicio} --> {final}\n{texto[1:] if texto[0] is " " else texto}\n\n'
    
    # mostramos el subtítulo
    print(subtitulo)
    
    
    # guardado del archivo .srt
    with open('../../files/transcripcion.srt', 'a', encoding='utf-8') as file:
        file.write(subtitulo)


#Uso de Modelo

# importamos los objetos procesador y modelo

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

procesador = AutoProcessor.from_pretrained(pretrained_model_name_or_path='openai/whisper-base')
# extracción de características del audio

caracteristicas = procesador(audio=audio, sampling_rate=16000, return_tensors='pt')

caracteristicas.keys()

# dimensiones del tensor de características

caracteristicas = caracteristicas.input_features

caracteristicas.shape

# carga del modelo 

modelo = AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_model_name_or_path='openai/whisper-base')


# generamos la respuesta del modelo

respuesta = modelo.generate(input_features=caracteristicas)

respuesta

# se devuelve la respuesta al procesador para obtener la respuesta en formato string

texto = procesador.batch_decode(respuesta, skip_special_tokens=True)

# para obtener el timestamp solo tenemos que pasar el argumento

respuesta = modelo.generate(input_features=caracteristicas, return_timestamps=True)

texto = procesador.batch_decode(respuesta, skip_special_tokens=True, decode_with_timestamps=True)