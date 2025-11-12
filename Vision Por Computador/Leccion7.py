#detectar objetos zero shot

# logging para quitar warnings de actualización de pesos del modelo

from transformers import logging

logging.set_verbosity_error()

# importamos desde la librería transformers el pipeline

from transformers import pipeline

# definimos la tarea y el modelo

tarea = 'zero-shot-object-detection'  

modelo = 'google/owlvit-base-patch32'

# iniciamos el clasificador de imagenes, los modelos se descargan en local, en este caso son unos 615Mb

img_pipe = pipeline(task=tarea, model=modelo)

# imagen de cachorros

url = 'https://i0.wp.com/planetamascotaperu.com/wp-content/uploads/2023/09/cachorros1.jpg'

# visualizamos la imagen con PIL en 600x400 píxeles

from PIL import Image
import requests as req

imagen = Image.open(req.get(url, stream=True).raw)

# etiquetas candidatas, ¿qué hay en la imagen?

etiquetas = ['parrot', 'cat', 'dog']

# llamada al pipeline para la clasificación de la imagen zero-shot

respuesta = img_pipe(image=imagen, candidate_labels=etiquetas)

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
    

# importamos los objetos procesador y modelo de detección de objetos 

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# transforma las imágenes 

procesador = AutoProcessor.from_pretrained('google/owlvit-base-patch32')

# imagen como objeto PIL, la cargamos de nuevo porque tenía cajas pintadas

imagen = Image.open(req.get(url, stream=True).raw)

# transformación de la imagen dándole las etiquetas que hemos definido

img_procesada = procesador(text=etiquetas, images=imagen, return_tensors='pt')

# una imagen, 3 canales, 768X768 píxeles

img_procesada['pixel_values'].shape

# mínimo y máximo de la imagen procesada

img_procesada['pixel_values'].min(), img_procesada['pixel_values'].max()

# importamos numpy

import numpy as np

# dimensión de la imagen original

np.array(imagen).shape

# mínimo y máximo de la imagen original

np.array(imagen).min(), np.array(imagen).max()

# tensor de salida del tokenizador, 3 etiquetas, 16 tokens de posición

img_procesada['input_ids'].shape

# inicializacion del modelo OwlViT

modelo_detector = AutoModelForZeroShotObjectDetection.from_pretrained('google/owlvit-base-patch32')

# resultado del modelo OwlViT al darle el vector

tensor = modelo_detector(**img_procesada)

# resultado post-procesado

resultado = procesador.post_process_object_detection(tensor, 
                                                     threshold=0.1, 
                                                     target_sizes=[imagen.size[::-1]])[0]


# hacemos un bucle sobre el resultado

for prob, eti, caja in zip(resultado['scores'], resultado['labels'], resultado['boxes']):
    
    # redondeamos y convertimos a lista los datos de los píxeles de la caja
    caja = [round(i, 2) for i in caja.tolist()]
    
    # creamos un print llamando al diccionario de etiquetas 
    print(f'Detectado {etiquetas[eti.item()]} con probabilidad {round(prob.item(), 3)} en {caja}')

# librerías
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection



def detector_zero_shot(imagen: object, etiquetas: list) -> list:
    
    """
    Función para detectar objetos en imagenes con modelo OwlViT Zero-Shot
    
    Params:
    imagen: objeto PIL.JpegImagePlugin.JpegImageFile
    etiquetas: lista de etiquetas definidas por el usuario
    
    Return:
    lista de strings que contienen la etiqueta, probabilidad y posiciones de las cajas
    """
    
    # transforma las imágenes 
    procesador = AutoProcessor.from_pretrained('google/owlvit-base-patch32')
    
    
    # transformación de la imagen dándole las etiquetas que hemos definido
    img_procesada = procesador(text=etiquetas, images=imagen, return_tensors='pt')
    
    
    # inicializacion del modelo OwlViT
    modelo_detector = AutoModelForZeroShotObjectDetection.from_pretrained('google/owlvit-base-patch32')
    
    
    # resultado del modelo OwlViT al darle el vector
    tensor = modelo_detector(**img_procesada)
    
    
    # resultado post-procesado
    resultado = procesador.post_process_object_detection(tensor, 
                                                         threshold=0.1, 
                                                         target_sizes=[imagen.size[::-1]])[0]
    
    
    # lista de salida de la función
    salida = []
    
    
    # hacemos un bucle sobre el resultado
    for prob, eti, caja in zip(resultado['scores'], resultado['labels'], resultado['boxes']):

        # redondeamos y convertimos a lista los datos de los píxeles de la caja
        caja = [round(i, 2) for i in caja.tolist()]

        # creamos un print llamanda al diccionario de etiquetas
        salida.append({'etiqueta': etiquetas[eti.item()],
                       'probabilidad': round(prob.item(), 3),
                       'caja': caja})
        
     
    return salida

detector_zero_shot(imagen, etiquetas)