#text to text / propm engeering

# para quitar warnings 

from transformers import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore')

# importamos librerías

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
from dotenv import load_dotenv      
import os 

# definimos la clase pipeline

class DifPipe:
    
    def __init__(self, 
                 modelo: str='dreamlike-art/dreamlike-photoreal-2.0',   # id del modelo
                 dispositivo: str='mps',                                # cpu, cuda, mps
                 pasos_inferencia: int=50,                              # pasos denoising
                 altura: int=512,                                       # altura imagen generada
                 ancho: int=512,                                        # ancho imagen generada
                 guia: float=8.5,                                       # ajuste a texto
                 jit: bool=True,                                        # compilación
                 safetensor: bool=True,                                 # tensores seguros
                 dtype: float=torch.float32                             # tipo de dato tensores
                ):
        
        
        # atributos
        self.modelo = modelo
        self.dispositivo = dispositivo
        self.pasos_inferencia = pasos_inferencia
        self.altura = altura
        self.ancho = ancho
        self.guia = guia
        self.jit = jit
        self.safetensor=safetensor
        self.dtype=dtype
        
        
        # importamos el token desde el archivo .env
        load_dotenv()
        HUGGINGFACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
        
        
        
        # inicialización del atributo pipeline
        self.pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=self.modelo,
                                                      use_safetensors=self.safetensor,
                                                      torch_dtype=self.dtype,            
                                                      token=HUGGINGFACE_TOKEN
                                                     ) 


        # selección del dispositivo
        self.pipe = self.pipe.to(device=self.dispositivo)


        # slicing de atención
        self.pipe.enable_attention_slicing()


        # scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(config=self.pipe.scheduler.config)
        
        
        
    def generate(self, prompt: str, negative_prompt: str='ugly') -> object:
        
        # generamos una imagen desde el prompt 
        resultado = self.pipe(prompt=prompt,                                # texto de entrada desde el usuario
                              negative_prompt=negative_prompt,              # lo que no queremos
                              num_inference_steps=self.pasos_inferencia,    # nº de pasos para inferencia
                              height=self.altura,                           # altura en píxeles
                              width=self.ancho,                             # ancho en píxeles
                              guidance_scale=self.guia,                     # más ajustado al texto
                              jit=self.jit,                                 # compilación dinámica
                             )
        
        torch.mps.empty_cache()
        
        return resultado.images[0]
    
modelo = 'stabilityai/stable-diffusion-2-1' 

#modelo = 'CompVis/stable-diffusion-v1-4'  # 4Gb

#modelo = 'SG161222/Realistic_Vision_V6.0_B1_noVAE'  # 5.5Gb

#modelo = 'stablediffusionapi/realistic-stock-photo'  # 7Gb

modelo = 'stablediffusionapi/realistic-stock-photo-v2' # 7Gb  # este mola

#modelo = 'digiplay/insaneRealistic_v1' # parece que hace blending

# inicialización del objeto

p = DifPipe(modelo=modelo, safetensor=None)