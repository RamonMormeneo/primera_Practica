#classificacion de imagen zero-shot

# logging para quitar warnings de actualización de pesos del modelo

from transformers import logging

logging.set_verbosity_error()

# importamos desde la librería transformers el pipeline

from transformers import pipeline

# definimos la tarea y el modelo

tarea = 'zero-shot-image-classification'  

modelo = 'google/siglip-so400m-patch14-384'

# iniciamos el clasificador de imagenes, los modelos se descargan en local, en este caso son unos 3.5Gb

img_pipe = pipeline(task=tarea, model=modelo)

# imagen de loros

url_loro = 'https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png'

# visualizamos la imagen con PIL en 600x400 píxeles

from PIL import Image
import requests as req

imagen = Image.open(req.get(url_loro, stream=True).raw)

imagen.resize((600, 400))

# etiquetas candidatas, ¿qué hay en la imagen?

etiquetas = ['animals', 'humans', 'landscape']

# llamada al pipeline para la clasificación de la imagen zero-shot

img_pipe(images=imagen, candidate_labels=etiquetas)

# nuevas etiquetas candidatas, ¿qué tipo de imagen es?

etiquetas = ['black and white', 'photorealist', 'painting']

# llamada al pipeline para la clasificación de la imagen zero-shot

img_pipe(images=imagen, candidate_labels=etiquetas)

#modelo classificador

# url de la imagen
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'


# imagen como objeto PIL
imagen = Image.open(req.get(url, stream=True).raw)

# visualización

imagen

# etiquetas propuestas 

etiquetas = ['cats', 'dogs']

# importamos los objetos procesador y modelo de clasificación de imagenes Zero-Shot

from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# transforma las imágenes 

procesador = AutoProcessor.from_pretrained('google/siglip-so400m-patch14-384')

# transformación de la imagen y de las etiquetas

img_txt_procesado = procesador(text=etiquetas, 
                               images=imagen.convert('RGB'), 
                               padding='max_length', 
                               return_tensors='pt')

# procesado de las etiquetas, 2 etiquetas, 64 tokens

img_txt_procesado['input_ids'].shape

# procesado de la imagen, 1 imagen, 3 canales, 384X384 píxeles

img_txt_procesado['pixel_values'].shape

# inicializacion del modelo Zero-Shot

modelo_zero_shot = AutoModelForZeroShotImageClassification.from_pretrained('google/siglip-so400m-patch14-384')

# resultado del modelo Zero-Shot al darle los vectores

tensor = modelo_zero_shot(**img_txt_procesado).logits_per_image

# tensor de salida

tensor

# predicción numérica del modelo

tensor.argmax().item()

# etiqueta predicha

etiquetas[tensor.argmax().item()]

# probabilidad de cada etiqueta utilizando la función sigmoide

tensor.sigmoid()

#resumen de codigo

# librerias
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification


def clasificador_zero_shot(imagen: object, etiquetas: list, modelo: str) -> str:
    
    
    # transforma las imágenes 
    procesador = AutoProcessor.from_pretrained(modelo)
    
    
    # transformación de la imagen y de las etiquetas
    img_txt_procesado = procesador(text=etiquetas, 
                                   images=imagen.convert('RGB'), 
                                   padding='max_length', 
                                   return_tensors='pt')
    
    
    # inicializacion del modelo Zero-Shot
    modelo_zero_shot = AutoModelForZeroShotImageClassification.from_pretrained(modelo)
    
    
    # resultado del modelo Zero-Shot al darle los vectores
    tensor = modelo_zero_shot(**img_txt_procesado).logits_per_image
    
    
    # etiqueta predicha
    resultado = etiquetas[tensor.argmax().item()]
    
    
    return resultado

clasificador_zero_shot(imagen, etiquetas, modelo)

#desde api

import os                           # librería del sistema operativo
from dotenv import load_dotenv      # para carga de las variables de entorno 


load_dotenv()


# token de hugging face
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

# url del modelo Zero-Shot

API_URL = 'https://api-inference.huggingface.co/models/google/siglip-so400m-patch14-384'

# ruta a la imagen

ruta = 'ficehros/tokyoGettyImages-1031467664.webp'

# lectura de la imagen

with open(ruta, 'rb') as f:
    data = f.read()

# etiquetas objetivo

etiquetas = ['street', 'landscape']

# creación json de consulta (base64 de datos binarios a una representación de texto ASCII)

import base64

json = {'parameters': {'candidate_labels': etiquetas},
        'inputs': base64.b64encode(data).decode('utf-8')}

# respuesta del modelo

respuesta = req.post(API_URL, headers=headers, json=json)

respuesta.json()

# resumen de la api


def consulta(ruta: str, etiquetas: list) -> list:
    
    """
    Función para realizar una consulta a la API de Hugging Face al modelo ViT
    
    Params:
    ruta: string, ruta al archivo de la imagen
    etiquetas: list, lista de strings con las etiquetas objetivo
    
    Return:
    lista de diccionarios con keys 'score' y 'label'
    """
    
    # variables globales token y url
    global HUGGING_FACE_TOKEN, API_URL
    
    
    # lectura de la imagen
    with open(ruta, 'rb') as f:
        data = f.read()
    
    
    # creación json de consulta
    json = {'parameters': {'candidate_labels': etiquetas},
            'inputs': base64.b64encode(data).decode('utf-8')}
    
    
    # respuesta de la api
    respuesta = req.post(API_URL, 
                         headers={'Authorization': f'Bearer {HUGGING_FACE_TOKEN}'}, 
                         json=json)
    
    
    return respuesta.json()

consulta(ruta, etiquetas)