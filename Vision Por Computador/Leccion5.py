#Deteccion de items

# logging para quitar warnings de actualización de pesos del modelo

from transformers import logging

logging.set_verbosity_error()

# importamos desde la librería transformers el pipeline

from transformers import pipeline

# definimos la tarea y el modelo

tarea = 'object-detection'  

modelo = 'facebook/detr-resnet-50'  

# iniciamos el clasificador de imagenes, los modelos se descargan en local, en este caso son unos 270Mb

pipe_50 = pipeline(task=tarea, model=modelo)

# el modelo 101

modelo = 'facebook/detr-resnet-101' 

# iniciamos el clasificador de imagenes, los modelos se descargan en local, en este caso son unos 410Mb

pipe_101 = pipeline(task=tarea, model=modelo)

# librerias PIL y requests
from PIL import Image
import requests as req


# url de la imagen
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'


# imagen como objeto PIL
imagen = Image.open(req.get(url, stream=True).raw)


# visualización
imagen

# inferencia modelo 50
respuesta = pipe_50(imagen)

respuesta

#Pintar cuadros

# importamos los objetos para pintar

from PIL import ImageDraw, ImageFont

# inicio del dibujo sobre la imagen

dibujo = ImageDraw.Draw(im=imagen)

# bucle para pintar todas las cajas y etiquetas

for e in respuesta:

    # texto 
    dibujo.text(xy=(e['box']['xmin'], e['box']['ymin']-20), 
                text=f"{e['label']}: {round(e['score'], 3)}",  
                fill=(0, 255, 0))
    
    # dibujo de un rectángulo 
    dibujo.rectangle(xy=[(e['box']['xmin'], e['box']['ymin']), (e['box']['xmax'], e['box']['ymax'])], 
                     outline=(0, 255, 0))
    
imagen

# importamos los objetos procesador y modelo de detección de objetos 

from transformers import DetrImageProcessor, DetrForObjectDetection

# transforma las imágenes 

procesador = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

# imagen como objeto PIL, la cargamos de nuevo porque tenía cajas pintadas

imagen = Image.open(req.get(url, stream=True).raw)

# transformación de la imagen

img_procesada = procesador(images=imagen, return_tensors='pt')

# una imagen, 3 canales, 800X1066 píxeles

img_procesada['pixel_values'].shape

# mínimo y máximo de la imagen procesada

img_procesada['pixel_values'].min(), img_procesada['pixel_values'].max()

# importamos numpy

import numpy as np

# dimensión de la imagen original

np.array(imagen).shape

# mínimo y máximo de la imagen original

np.array(imagen).min(), np.array(imagen).max()

# inicializacion del modelo DETR

modelo_detector = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

# resultado del modelo DETR al darle el vector

tensor = modelo_detector(**img_procesada)

# dimensión del tensor de salida, una imagen, 100 características y 92 clases

tensor.logits.shape

# resultado post-procesado

resultado = procesador.post_process_object_detection(tensor, 
                                                     threshold=0.8, 
                                                     target_sizes=[imagen.size[::-1]])[0]

resultado

# diccionario de etiquetas

etiquetas = modelo_detector.config.id2label

print(etiquetas)

# hacemos un bucle sobre el resultado

for prob, eti, caja in zip(resultado['scores'], resultado['labels'], resultado['boxes']):
    
    # redondeamos y convertimos a lista los datos de los píxeles de la caja
    caja = [round(i, 2) for i in caja.tolist()]
    
    # creamos un print llamanda al diccionario de etiquetas
    print(f'Detectado {etiquetas[eti.item()]} con probabilidad {round(prob.item(), 3)} en {caja}')

#resumen de codigo

# librerias
from transformers import DetrImageProcessor, DetrForObjectDetection



def detector_objetos(imagen: object) -> list:
    
    """
    Función para detectar objetos en imagenes con modelo DETR
    
    Params:
    imagen: objeto PIL.JpegImagePlugin.JpegImageFile
    
    Return:
    lista de strings que contienen la etiqueta, probabilidad y posiciones de las cajas
    """
    
    
    # transforma las imágenes 
    procesador = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
    
    
    # transformación de la imagen
    img_procesada = procesador(images=imagen, return_tensors='pt')
    
    
    # inicializacion del modelo DETR
    modelo_detector = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
    
    
    # resultado del modelo DETR al darle el vector
    tensor = modelo_detector(**img_procesada)
    
    
    # resultado post-procesado
    resultado = procesador.post_process_object_detection(tensor, 
                                                         threshold=0.8, 
                                                         target_sizes=[imagen.size[::-1]])[0]
    
    
    # diccionario de etiquetas
    etiquetas = modelo_detector.config.id2label
    
    
    # lista de salida de la función
    salida = []
    
    
    # hacemos un bucle sobre el resultado
    for prob, eti, caja in zip(resultado['scores'], resultado['labels'], resultado['boxes']):

        # redondeamos y convertimos a lista los datos de los píxeles de la caja
        caja = [round(i, 2) for i in caja.tolist()]

        # creamos un print llamanda al diccionario de etiquetas
        salida.append(f'Detectado {etiquetas[eti.item()]} con probabilidad {round(prob.item(), 3)} en {caja}')
        
     
    return salida

detector_objetos(imagen)

#uso desde api

import os                           # librería del sistema operativo
from dotenv import load_dotenv      # para carga de las variables de entorno 


load_dotenv()


# token de hugging face
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

# url del modelo DETR

API_URL = 'https://api-inference.huggingface.co/models/facebook/detr-resnet-50'

# autenticación

headers = {'Authorization': f'Bearer {HUGGING_FACE_TOKEN}'}

# ruta a la imagen

ruta = 'ficehros/tokyoGettyImages-1031467664.webp'

# lectura de la imagen

with open(ruta, 'rb') as f:
    data = f.read()

# respuesta al modelo

respuesta = req.post(API_URL, headers=headers, data=data)

respuesta.json()

#resumen
# resumen de la api


def consulta(ruta: str) -> list:
    
    """
    Función para realizar una consulta a la API de Hugging Face al modelo DETR
    
    Params:
    ruta: string, ruta al archivo de la imagen
    
    Return:
    lista de diccionarios con keys 'score', 'label' y 'box'
    """
    
    # variables globales token y url
    global HUGGING_FACE_TOKEN, API_URL
    
    
    # lectura de la imagen
    with open(ruta, 'rb') as f:
        data = f.read()
        
    
    # respuesta de la api
    respuesta = req.post(API_URL, 
                         headers={'Authorization': f'Bearer {HUGGING_FACE_TOKEN}'}, 
                         data=data)
    
    
    return respuesta.json()

consulta(ruta)