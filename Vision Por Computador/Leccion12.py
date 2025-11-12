#image to text

# importamos desde la librería transformers el pipeline

from transformers import pipeline

# definimos la tarea y el modelo

tarea = 'image-to-text'  

modelo = 'Salesforce/blip-image-captioning-large'

# iniciamos el descriptor de imagenes, los modelos se descargan en local, en este caso son unos 2Gb

img_pipe = pipeline(task=tarea, model=modelo, max_new_tokens=100)

img_pipe('ficehros/tokyoGettyImages-1031467664.webp')


#uso de modelo

# importamos los objetos procesador y modelo descriptor de imagenes

from transformers import BlipProcessor, BlipForConditionalGeneration

# transforma las imágenes y el texto

procesador = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large')

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 

# visualizamos la imagen con PIL en 400x400 píxeles

from PIL import Image
import requests as req

imagen = Image.open(req.get(img_url, stream=True).raw)

imagen.resize((600, 400))

# transformación de la imagen y el texto, al añadir este texto se pone una condición

img_txt_procesado = procesador(images=imagen, text='a photography of', return_tensors='pt')

# una imagen, 3 canales, 384x384 píxeles

img_txt_procesado['pixel_values'].shape

# tokenizacion texto

img_txt_procesado['input_ids']

# inicializacion del modelo Blip

modelo_descriptor = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large')

# resultado del modelo Blip al darle el vector

tensor = modelo_descriptor.generate(**img_txt_procesado)

# respuesta del modelo

procesador.decode(tensor[0], skip_special_tokens=True)

#forma no condicional 
# transformación de la imagen 
img_txt_procesado = procesador(images=imagen, return_tensors='pt')


# resultado del modelo Blip al darle el vector
tensor = modelo_descriptor.generate(**img_txt_procesado)


# respuesta del modelo
procesador.decode(tensor[0], skip_special_tokens=True)