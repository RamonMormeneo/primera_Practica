#texto a video

# para quitar warnings 

from transformers import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore')

# importamos las librerias torch, el pipeline y el scheduler de diffusers

import torch  
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

# definimos el modelo

modelo = 'cerspense/zeroscope_v2_576w'

# inicializamos el pipeline

pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=modelo, 
                                         torch_dtype=torch.float16)

# cambio de dispositivo y añadido del scheduler al pipeline

pipe.to('cuda')

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# texto para generar el video

prompt = 'Darth Vader is surfing on waves with a light saber in his hand'

# generación del video frame a frame como array de numpy

frames = pipe(prompt=prompt, 
              num_inference_steps=40, 
              height=320, 
              width=576, 
              num_frames=24).frames

# herramienta para exportar el video, lo guarda y devuelve la ruta

from diffusers.utils import export_to_video

video = export_to_video(frames[0])

# descarga del archivo desde colab

from google.colab import files

files.download(video) 

# liberación de memoria

torch.cuda.empty_cache()

del pipe, frames, video