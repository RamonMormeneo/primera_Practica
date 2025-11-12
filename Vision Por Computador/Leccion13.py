#Imagen a Imagen

# para quitar warnings 

from transformers import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore')

# importamos desde la librería diffusers el pipeline

from diffusers import DiffusionPipeline

# definimos el modelo

modelo = 'timbrooks/instruct-pix2pix' 

# importamos torch y el scheduler

import torch
from diffusers import EulerAncestralDiscreteScheduler

# iniciamos el modelo de difusión, los modelos se descargan en local, en este caso son unos 5.5Gb

pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=modelo, 
                                         torch_dtype=torch.float16, 
                                         safety_checker=None)

# cambio de dispositivo y añadido del scheduler al pipeline

pipe.to('mps')

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# url de la imagen

url = 'https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg'

# visualizamos la imagen con PIL en 400x400 píxeles

from PIL import Image
import requests as req

imagen = Image.open(req.get(url, stream=True).raw)

imagen.resize((400, 400))

# texto de referencia

prompt = 'turn him into cyborg'

# llamada al pipeline

resultado = pipe(prompt=prompt, 
                 image=imagen, 
                 num_inference_steps=50, 
                 image_guidance_scale=1, 
                 guidance_scale=7.5)

resultado

#generador de qr


# importamos la librería

import qrcode

# creamos la imagen del qr con la url de una web

imagen_qr = qrcode.make('https://www.hackio.com/')

# la guardamos y la visualizamos

imagen_qr.save('../../images/imagen_qr.png')

imagen_qr

# modelo control

from diffusers import ControlNetModel

red_control = ControlNetModel.from_pretrained('vertxlabs/controlnet_qrcode-control_v11p_v1')

# modelo de difusión

from diffusers import StableDiffusionControlNetPipeline

modelo = 'stabilityai/stable-diffusion-2-1'

pipe = StableDiffusionControlNetPipeline.from_pretrained(pretrained_model_name_or_path=modelo,
                                                         controlnet=red_control)

# cambio de dispositivo y añadido del scheduler al pipeline

pipe.to('mps')

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# imagen del QR

imagen_qr = Image.open('../../images/imagen_qr.png')

# textos

prompt = '''Blooming chinese chrysanthemum, 
            green leaves growing wantonly, flowers, 
            Complex patterns on the border,
            Masterpiece Art, Beauty, 8K,  Unreal Engine'''


negative_prompt = 'ugly, low quality, blurry, nsfw'

# llamada al pipeline

resultado = pipe(prompt=prompt,
                 negative_prompt=negative_prompt, 
                 image=imagen_qr,
                 control_image=imagen_qr,
                 width=512,
                 height=512,
                 guidance_scale=20,
                 controlnet_conditioning_scale=2.,
                 strength=0.9, 
                 num_inference_steps=50,
                )

resultado.images[0]