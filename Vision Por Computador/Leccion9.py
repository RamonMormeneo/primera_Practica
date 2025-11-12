#estimacion de preofundidad

# logging para quitar warnings de actualización de pesos del modelo

from transformers import logging

logging.set_verbosity_error()

# importamos desde la librería transformers el pipeline

from transformers import pipeline

# definimos la tarea y el modelo

tarea = 'depth-estimation'  

modelo = 'Intel/dpt-large' 

# iniciamos el estimador de profundidad, los modelos se descargan en local, en este caso son unos 1.4GB

prof_pipe = pipeline(task=tarea, model=modelo)

# url de la imagen

url = 'https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640'

# librerias PIL y requests
from PIL import Image
import requests as req


# imagen como objeto PIL
imagen = Image.open(fp=req.get(url, stream=True).raw)

# llamada al pipeline para la segmentación de la imagen

respuesta = prof_pipe(images=imagen)

respuesta.keys()

# estimación distancia mínima en metros

respuesta['predicted_depth'].min()

# estimación distancia máxima en metros

respuesta['predicted_depth'].max()

# visualización de la estimación de profundidad

respuesta['depth']

#uso de modelo

# importamos los objetos procesador y modelo de estimación de profundidad

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# procesa las imágenes y extrae las características

procesador = AutoImageProcessor.from_pretrained('Intel/dpt-large')

# procesamiento de la imagen

img_procesada = procesador(images=imagen, return_tensors='pt')

# una imagen, 3 canales, 384X384 píxeles

img_procesada['pixel_values'].shape

# inicializacion del modelo DPT

modelo_estimador = AutoModelForDepthEstimation.from_pretrained('Intel/dpt-large')

# resultado del modelo GLPN al darle la imagen procesada

tensor = modelo_estimador(**img_procesada)


# dimensiones del tensor

tensor.predicted_depth.size()

# estimación distancia mínima en metros

tensor.predicted_depth.min()

# estimación distancia máxima en metros

tensor.predicted_depth.max()

# conversión del tensor a array de numpy
profundidad = tensor.predicted_depth[0].detach().numpy()

# normalización, valores entre 0 y 255
array = (profundidad * 255 / profundidad.max()).astype('uint8')

# imagen como objeto PIL
Image.fromarray(array)

#resumen

def estimador_profundidad(imagen: object) -> dict:
    
    """
    Función para estimar la profundidad de una imagen
    
    Params:
    imagen: objeto PIL.JpegImagePlugin.JpegImageFile
    
    Return:
    diccionario con el tensor de estimación de profundidad en metros y la imagen como objeto PIL
    """
    
    
    # procesa las imágenes y extrae las características
    procesador = AutoImageProcessor.from_pretrained('Intel/dpt-large')
    
    
    # procesamiento de la imagen
    img_procesada = procesador(images=imagen, return_tensors='pt')
    
    # inicializacion del modelo DPT
    modelo_estimador = AutoModelForDepthEstimation.from_pretrained('Intel/dpt-large')
    
    
    # resultado del modelo GLPN al darle la imagen procesada
    tensor = modelo_estimador(**img_procesada)
    
    
    # conversión del tensor a array de numpy
    profundidad = tensor.predicted_depth[0].detach().numpy()

    # normalización, valores entre 0 y 255
    array = (profundidad * 255 / profundidad.max()).astype('uint8')

    # imagen como objeto PIL
    img_profundidad = Image.fromarray(array)
    
    return {'tensor': tensor.predicted_depth[0], 'imagen': img_profundidad}

res = estimador_profundidad(imagen)