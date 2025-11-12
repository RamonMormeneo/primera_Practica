#segmentacion de imagen

# logging para quitar warnings de actualización de pesos del modelo

from transformers import logging

logging.set_verbosity_error()

# importamos desde la librería transformers el pipeline

from transformers import pipeline

# definimos la tarea y el modelo

tarea = 'image-segmentation'  

modelo = 'facebook/maskformer-swin-base-coco'  

# iniciamos el segmentador de imagenes, los modelos se descargan en local, en este caso son unos 415MB

seg_pipe = pipeline(task=tarea, model=modelo)

# url de la imagen

url = 'https://i0.wp.com/planetamascotaperu.com/wp-content/uploads/2023/09/cachorros1.jpg'

# librerias PIL y requests
from PIL import Image
import requests as req


# imagen como objeto PIL
imagen = Image.open(fp=req.get(url, stream=True).raw)

# llamada al pipeline para la segmentación de la imagen

respuesta = seg_pipe(images=imagen)

respuesta

# máscara

respuesta[0]['mask']

# importamos las operaciones de PIL
from PIL import ImageOps

# hacemos una copia de la imagen
im = imagen.copy()

# ponemos en negativo la mascara
mascara = ImageOps.invert(image=respuesta[0]['mask'])

# combinamos ambas imágenes
im.paste(im=mascara, mask=mascara)

im

#se puedne cobinar varias mascaras a la vez
# importamos numpy
import numpy as np

# unimos varias máscaras
mascara = Image.fromarray(obj=np.array(respuesta[0]['mask'])+np.array(respuesta[3]['mask']))

# hacemos una copia de la imagen
im = imagen.copy()

# ponemos en negativo la mascara
mascara = ImageOps.invert(image=mascara)

# combinamos ambas imágenes
im.paste(im=mascara, mask=mascara)

im

#uso del modelo

# importamos los objetos procesador y modelo de segmentación de imagenes

from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation

# procesa las imágenes y extrae las características

procesador = MaskFormerImageProcessor.from_pretrained('facebook/maskformer-swin-base-coco')

# procesamiento de la imagen

img_procesada = procesador(images=imagen, return_tensors='pt')

# tensores en la imagen procesada

img_procesada.keys()

# una imagen, 3 canales, 800x1152 píxeles

img_procesada['pixel_values'].shape

# una máscara de 800x1152 píxeles

img_procesada['pixel_mask'].shape

# inicializacion del modelo maskformer

modelo_segmentador = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-base-coco')

# resultado del modelo maskformer al darle la imagen procesada

tensor = modelo_segmentador(**img_procesada)

# post-procesamiento, devuelve máscaras y etiquetas

segmentacion = procesador.post_process_panoptic_segmentation(tensor, target_sizes=[imagen.size[::-1]])[0]

# tensor de máscaras

mascaras = segmentacion['segmentation']

# máximo y mínimo

mascaras.max(), mascaras.min()

# dimensión de las máscaras

mascaras.size()

# dimensión de la imagen original

imagen.size
# prediccion de etiquetas y probabilidad asociada

predicciones = segmentacion['segments_info']


# máscara del gato

gato = np.array(np.where(mascaras==2, 0, 255)).astype(np.uint8)

gato = Image.fromarray(gato)

# filtramos la imagen original

# hacemos una copia de la imagen
im = imagen.copy()

# combinamos ambas imágenes
im.paste(im=gato, mask=gato)

# todas las etiquetas

etiquetas = modelo_segmentador.config.id2label

print(etiquetas)

# etiqueta 15

etiquetas[15]

# etiqueta 116

etiquetas[116]

#resumen

# librerías

import numpy as np
from PIL import Image
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation



def segmentador_imagen(imagen: object) -> list:
    
    """
    Función para segmentar imagenes con modelo maskformer
    
    Params:
    imagen: objeto PIL.JpegImagePlugin.JpegImageFile
    
    Return:
    lista de diccionarios con keys etiquetas, probabilidades y máscaras 
    """
    
    
    # procesa las imágenes y extrae las características
    procesador = MaskFormerImageProcessor.from_pretrained('facebook/maskformer-swin-base-coco')
    
    
    # procesamiento de la imagen
    img_procesada = procesador(images=imagen, return_tensors='pt')
    
    
    # inicializacion del modelo maskformer
    modelo_segmentador = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-base-coco')
    
    
    # resultado del modelo maskformer al darle la imagen procesada
    tensor = modelo_segmentador(**img_procesada)
    
    
    # post-procesamiento, devuelve máscaras y etiquetas
    segmentacion = procesador.post_process_panoptic_segmentation(tensor, target_sizes=[imagen.size[::-1]])[0]
    
    
    # tensor de máscaras
    mascaras = segmentacion['segmentation']
    
    
    # prediccion de etiquetas y probabilidad asociada
    predicciones = segmentacion['segments_info']

    
    # todas las etiquetas
    etiquetas = modelo_segmentador.config.id2label
    
    
    # lista de resultado
    resultado = []
    
    for i in range(len(predicciones)):
        
        etiqueta = etiquetas[predicciones[i]['label_id']]
        
        prob = predicciones[i]['score']
        
        mascara = mascaras[i]
        
        mascara = np.array(np.where(mascaras==i+1, 0, 255)).astype(np.uint8)
        
        mascara = Image.fromarray(mascara)
        
        dictio = {'etiqueta': etiqueta, 'probabilidad': prob, 'mascara': mascara}
        
        resultado.append(dictio)
        
    
    return resultado
    
res = segmentador_imagen(imagen)

res[1]['mascara']