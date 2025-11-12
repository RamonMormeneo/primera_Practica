#Classificacion de imagen
# importamos desde la librería transformers el pipeline

from transformers import pipeline

# definimos la tarea y el modelo

tarea = 'image-classification'  

modelo = 'google/vit-base-patch16-224'

# iniciamos el clasificador de imagenes, los modelos se descargan en local, en este caso son unos 350Mb

img_pipe = pipeline(task=tarea, model=modelo)

# llamada al pipeline con imagen en local

img_pipe('ficehros/tokyoGettyImages-1031467664.webp')

# imagen de gato

url_gato = 'https://www.purina.es/sites/default/files/styles/square_medium_440x440/public/2024-02/sitesdefaultfilesstylessquare_medium_440x440public2022-06Siamese-Cat_0.jpg?itok=SpejPfbc'

# visualizamos la imagen con PIL en 400x400 píxeles

from PIL import Image
import requests as req

imagen = Image.open(req.get(url_gato, stream=True).raw)

imagen.resize((400, 400))

# llamada al pipeline para la clasificación de la imagen

img_pipe(url_gato)

# imagen de fuego

url_fuego = 'https://i.pinimg.com/736x/1c/ee/b6/1ceeb6c650802b171763c969bc3a9a79.jpg'

# visualizamos la imagen con PIL en 400x400 píxeles

imagen = Image.open(req.get(url_fuego, stream=True).raw)

imagen.resize((400, 400))

# llamada al pipeline para la clasificación de la imagen

img_pipe(imagen)

#uso del modelo classificador
# importamos los objetos procesador y modelo de clasificación de imagenes

from transformers import ViTImageProcessor, ViTForImageClassification

# transforma las imágenes 

procesador = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# transformación de la imagen

img_procesada = procesador(images=imagen, return_tensors='pt')

# mínimo y máximo de la imagen procesada

img_procesada['pixel_values'].min(), img_procesada['pixel_values'].max()

# importamos numpy

import numpy as np

# dimensión de la imagen original

np.array(imagen).shape

# mínimo y máximo de la imagen original

np.array(imagen).min(), np.array(imagen).max()

# inicializacion del modelo ViT

modelo_vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# resultado del modelo Vit al darle el vector

tensor = modelo_vit(**img_procesada).logits


# dimensión del tensor de salida

tensor.shape

# predicción numérica del modelo

tensor.argmax().item()

# diccionario de etiquetas

etiquetas = modelo_vit.config.id2label

# etiqueta predicha

etiquetas[tensor.argmax().item()]

# índices de las 10 etiquetas más probables, de menor a mayor

indices_probables = tensor[0].sort().indices[-10:]


# 10 etiquetas más probables, de mayor a menor

etiquetas_probables = [etiquetas[i.item()] for i in indices_probables][::-1]

#resumen de codigo


def clasificador_imagen(imagen: object) -> list:
    
    """
    Función para clasificar imagenes con modelo ViT
    
    Params:
    imagen: objeto PIL.JpegImagePlugin.JpegImageFile
    
    Return:
    lista con las 10 etiquetas más probables
    """
    
    # transforma las imágenes 
    procesador = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # transformación de la imagen
    img_procesada = procesador(images=imagen, return_tensors='pt')
    
    # inicializacion del modelo ViT
    modelo_vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    # resultado del modelo Vit al darle el vector
    tensor = modelo_vit(**img_procesada).logits
    
    # diccionario de etiquetas
    etiquetas = modelo_vit.config.id2label
    
    
    # índices de las 10 etiquetas más probables, de menor a mayor
    indices_probables = tensor[0].sort().indices[-10:]
    
    # 10 etiquetas más probables, de mayor a menor
    etiquetas_probables = [etiquetas[i.item()] for i in indices_probables][::-1]

    return etiquetas_probables

clasificador_imagen(imagen)

#uso del modelo desde la api

# resumen de la api
import os                           # librería del sistema operativo
from dotenv import load_dotenv      # para carga de las variables de entorno 


load_dotenv()


# token de hugging face
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
# url del modelo ViT

API_URL = 'https://api-inference.huggingface.co/models/google/vit-base-patch16-224'

def consulta(ruta: str) -> list:
    
    """
    Función para realizar una consulta a la API de Hugging Face al modelo ViT
    
    Params:
    ruta: string, ruta al archivo de la imagen
    
    Return:
    lista de diccionarios con keys 'score' y 'label'
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

# ruta a la imagen

ruta = 'ficehros/tokyoGettyImages-1031467664.webp'

consulta(ruta)

