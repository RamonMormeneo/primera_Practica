#pipline

# importamos desde la librería transformers AutoProcessor y AutoModelForPreTraining

from transformers import AutoProcessor, AutoModelForPreTraining

# definimos el modelo

modelo = 'HuggingFaceM4/tiny-random-idefics'

# definicion del procesador

procesador = AutoProcessor.from_pretrained(modelo)

# imagen

from PIL import Image
import requests as req


url_img = 'https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG'

respuesta = req.get(url=url_img, stream=True).raw

imagen = Image.open(fp=respuesta)

# texto 

texto = 'In this picture from Asterix and Obelix, we can see'

# definimos el prompt con la imagen y el texto, pueden ser varios

prompts = [[url_img, texto],]


# tensor de salida del procesador

tensor = procesador(prompts, return_tensors='pt').to('cpu')

tensor.input_ids.shape

# definicion del modelo

modelo_multi = AutoModelForPreTraining.from_pretrained(modelo) 

# descripcion del modelo

modelo_multi