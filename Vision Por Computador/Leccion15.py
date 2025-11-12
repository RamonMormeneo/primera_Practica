#Imagen a video
# para quitar warnings 

from transformers import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore')

# importamos las librerias

from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from PIL import Image
import requests as req

# url de la imagen

url = 'https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg'

# visualización

imagen = Image.open(req.get(url, stream=True).raw)

imagen.resize((400, 400))

# definición del modelo

modelo = 'vdo/stable-video-diffusion-img2vid-xt-1-1'

# inicializamos el pipeline

pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=modelo,
                                             torch_dtype=torch.float16,
                                             use_safetensors=True)

# cambio de dispositivo y añadido del scheduler al pipeline

pipeline.to('cuda')

pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

# descarga los pesos a la CPU y los cargar en la GPU solo cuando se realiza la pasada hacia adelante 

pipeline.enable_sequential_cpu_offload()

# llamada al pipeline, el cambio en dimensiones también baja el nivel de memoria necesaria

frames = pipeline(image=imagen, 
                  num_inference_steps=60,
                  height=512,
                  width=512).frames

# herramienta para exportar el video, lo guarda y devuelve la ruta

from diffusers.utils import export_to_video

video = export_to_video(frames[0])

# descarga del archivo desde colab

from google.colab import files

files.download(video) 