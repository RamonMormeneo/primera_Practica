#de texto a imagen

# importamos desde la librería diffusers el pipeline

from diffusers import DiffusionPipeline

# definimos el modelo

modelo = 'stabilityai/stable-diffusion-2-1' 

# iniciamos el modelo de difusión, los modelos se descargan en local, en este caso son unos 5.15Gb

pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=modelo)

# texto que le pasamos al modelo

prompt = 'realistic Cheshire cat with a witch hat'

# generamos una imagen desde el prompt con un solo paso

resultado = pipe(prompt=prompt, num_inference_steps=1)

# imagen generada

imagen = resultado.images[0]

# tipo de dato de la imagen

type(imagen)

# importamos torch

import torch

# iniciamos el modelo de difusión

pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=modelo, # nombre del modelo
                                         torch_dtype=torch.float16,            # tipo de dato del tensor
                                         use_safetensors=True                  # mejora eficiencia y seguridad
                                        )

# cambiamos el hardware usado a mps

pipe = pipe.to(device='mps') 

# slicing de atención

pipe.enable_attention_slicing()  

# llamada al pipeline

resultado = pipe(prompt=prompt,                   # texto de entrada desde el usuario
                 negative_prompt='ugly, art',     # lo que no queremos
                 num_inference_steps=50,          # nº de pasos para inferencia
                 height=600,                      # altura en píxeles
                 width=600,                       # ancho en píxeles
                 guidance_scale=10,               # más ajustado al texto
                 num_images_per_prompt=2,         # genera 2 imágenes
                 jit=True,                        # compilación dinámica
                )



# imagen generada

imagen = resultado.images[0]

#uso del modelo

# importamos librerías
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel

# definimos los modelos

modelo_difusion = 'CompVis/stable-diffusion-v1-4'

modelo_texto = 'openai/clip-vit-large-patch14'

# cargamos el VAE

vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=modelo_difusion,
                                    subfolder='vae')  

# cargamos el modelo de texto, incluyendo el tokenizador

tokenizador = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path=modelo_texto)  
encoder_texto = CLIPTextModel.from_pretrained(pretrained_model_name_or_path=modelo_texto)

# cargamos la U-Net

unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path=modelo_difusion,
                                            subfolder='unet')

# ajustamos el dispositivo de nuestra máquina

dispositivo = 'mps'

vae = vae.to(dispositivo)

encoder_texto = encoder_texto.to(dispositivo)

unet = unet.to(dispositivo) 

# importamos torch

import torch

# texto para generar la imagen
prompt = ['a close up of a fire breathing pokemon figure, digital art']

# dimensiones de la imagen de salida en píxeles
altura = 512                         
ancho = 512        

# nº de pasos para inferencia
num_inference_steps = 100           

# ajuste al texto
guidance_scale = 7.5                

# generador aleatorio
generador = torch.manual_seed(42)    

# tamaño del lote
batch_size = len(prompt)

# generamos los tokens
tokens = tokenizador(text=prompt, 
                     padding='max_length', 
                     max_length=tokenizador.model_max_length, 
                     truncation=True, 
                     return_tensors='pt')



# se generan los embeddings
embeddings_texto = encoder_texto(input_ids=tokens.input_ids.to(dispositivo))[0]

# generamos los tokens del texto vacío
token_incond = tokenizador(text=[''] * batch_size, 
                           padding='max_length', 
                           max_length=tokens.input_ids.shape[-1], 
                           return_tensors='pt')


# se generan los embeddings del texto vacío
embeddings_texto_incond = encoder_texto(input_ids=token_incond.input_ids.to(dispositivo))[0]  

# se concatenan ambos embeddings

embeddings_texto = torch.cat([embeddings_texto_incond, embeddings_texto])

# tensor de ruido
latentes = torch.randn(size=(batch_size, unet.config.in_channels, altura // 8, ancho // 8),
                       generator=generador)


latentes = latentes.to(dispositivo)

# dimensiones del espacio latente

latentes.size()

# importamos el scheduler
from diffusers import LMSDiscreteScheduler


# se inicia el scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085,  # cantidad de ruido añadido en los primeros pasos
                                 beta_end=0.012,      # cantidad de ruido añadido en los últimos pasos
                                 beta_schedule='scaled_linear',  # programación lineal escalada
                                 num_train_timesteps=1000        # número total de pasos
                                )


# se configura el scheduler con el número de pasos de inferencia 
scheduler.set_timesteps(num_inference_steps)

# desviación del ruido inicial

latentes = latentes * scheduler.init_noise_sigma

# para barra de progreso
from tqdm.auto import tqdm


for paso in tqdm(scheduler.timesteps):
    
    # expande los latentes para evitar hacer dos pasadas hacia adelante.
    modelo_latente = torch.cat([latentes] * 2)
    
    # paso al scheduler
    modelo_latente = scheduler.scale_model_input(modelo_latente, timestep=paso)
    
    
    # predice el ruido residual 
    with torch.no_grad():
        pred_ruido = unet(modelo_latente, paso, encoder_hidden_states=embeddings_texto).sample

    # aplicar guidance
    ruido_incond, ruido_texto = pred_ruido.chunk(2)
    pred_ruido = ruido_incond + guidance_scale * (ruido_texto - ruido_incond)

    # calcula la muestra de ruido previo x_t -> x_t-1
    latentes = scheduler.step(pred_ruido, paso, latentes).prev_sample

    # constante empírica de reescalado

cte = 1 / 0.18215

# se introduce el espacio latente en el VAE

imagen = vae.decode(cte*latentes).sample

# dimensiones imagen de salida, 3 canales, 512x512

imagen.size()

# importamos PIL
from PIL import Image

# # normalización [0,1]
imagen = (imagen / 2 + 0.5).clamp(0, 1)

# cambio de dimensiones y conversión a array de numpy
imagen = imagen.detach().cpu().permute(0, 2, 3, 1).numpy()

# rango [0,1] a [0,255]
imagen = (imagen * 255).round().astype('uint8')

# conversión a objeto PIL
imagen = Image.fromarray(imagen[0])

# borrado de memoria

del tokens, tokenizador, vae, unet, pred_ruido, ruido_incond, ruido_texto, embeddings_texto_incond

del modelo_latente, encoder_texto, embeddings_texto, imagen, latentes, scheduler

torch.mps.empty_cache()