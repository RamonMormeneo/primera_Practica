#pillow y su uso

from PIL import Image

url_img = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Great_Wave_off_Kanagawa2.jpg/600px-Great_Wave_off_Kanagawa2.jpg'

# con requests vamos a pedir el recurso

import requests as req

# objeto imagen desde la web, stream=True es para bajarlo por lotes (chunks)

respuesta = req.get(url=url_img, stream=True).raw

# abrimos la imagen existente en la web

imagen = Image.open(fp=respuesta)

# ruta donde vamos a guardar la imagen

ruta_guardado = 'ficehros/tokyoGettyImages-1031467664.webp'

# guardado de imagen

imagen.save(fp=ruta_guardado)

#Manipular imagen
# nuevas dimensiones de la imagen (x, y)

nuevo_tamaño = (400, 200)

# cambio de tamaño

imagen.resize(size=nuevo_tamaño)

# caja de recorte (izq, arriba, dcha, abajo) en pixeles

caja = (0, 0, 100, 150)

# recorte de la firma

imagen.crop(box=caja)

# ejemplo de la ola (310, 250)

caja = (70, 0, 380, 250)

imagen.crop(box=caja)

# rotación, antihorario


grados = 90


imagen.rotate(angle=grados)

# transposición eje y

imagen.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)

# transposición eje x

imagen.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)

# tupla de transformación (a, b, c, d, e, f)

transformacion = (1, 0, -100, 0, 1, 10)    # 100 a la dcha , 10 arriba

# traslación

imagen.transform(size=imagen.size, method=Image.AFFINE, data=transformacion)

# filtros de imagen desde PIL

from PIL import ImageFilter

# desenfoque, blur

imagen.filter(ImageFilter.BLUR)

# desenfoque gaussiano

imagen.filter(ImageFilter.GaussianBlur(radius=10))

# contorno de la imagen

imagen.filter(ImageFilter.CONTOUR)

# suavizado

imagen.filter(ImageFilter.SMOOTH)

# más suavizado

imagen.filter(ImageFilter.SMOOTH_MORE)

#Dibujo en imagern

# importamos la clase

from PIL import ImageDraw

# petición de la imagen
respuesta = req.get(url=url_img, stream=True).raw

# conversión de la imagen a objeto PIL
imagen = Image.open(fp=respuesta)

# inicio del dibujo sobre la imagen
dibujo = ImageDraw.Draw(im=imagen)

# dibujo de una línea, desde (0,0) hasta el tamaño de la imagen en blanco por defecto
dibujo.line(xy=(0, 0) + imagen.size)

# dibujo de una línea, desde (0,N2) hasta (N1,0) en azul y ancho 3
dibujo.line(xy=(0, imagen.size[1], imagen.size[0], 0), fill=(0, 0, 255), width=3)

# el dibujo cambia el objeto imagen
imagen

# petición de la imagen
respuesta = req.get(url=url_img, stream=True).raw

# conversión de la imagen a objeto PIL
imagen = Image.open(fp=respuesta)

# inicio del dibujo sobre la imagen
dibujo = ImageDraw.Draw(im=imagen)

# dibujo de un rectángulo desde el punto (300, 300) hasta el tamaño de la imagen en azul
dibujo.rectangle(xy=[(300, 300), imagen.size], fill=(0, 0, 255))

# el dibujo cambia el objeto imagen
imagen

# elipse

# petición de la imagen
respuesta = req.get(url=url_img, stream=True).raw

# conversión de la imagen a objeto PIL
imagen = Image.open(fp=respuesta)

# inicio del dibujo sobre la imagen
dibujo = ImageDraw.Draw(im=imagen)

# dibujo de una elipse desde el punto (300, 300) hasta el tamaño de la imagen en azul
dibujo.ellipse(xy=[(300, 300), imagen.size], fill=(0, 0, 255))

# el dibujo cambia el objeto imagen
imagen

# circulo

# petición de la imagen
respuesta = req.get(url=url_img, stream=True).raw

# conversión de la imagen a objeto PIL
imagen = Image.open(fp=respuesta)

# inicio del dibujo sobre la imagen
dibujo = ImageDraw.Draw(im=imagen)

# dibujo de un círculo desde el punto (400, 50) hasta (500, 150) en rojo, 100X100
dibujo.ellipse(xy=[(400, 50), (500, 150)], fill=(255, 0, 0), width=500)

imagen

#Texto

# objecto para el usa de las fuentes

from PIL import ImageFont

# petición de la imagen
respuesta = req.get(url=url_img, stream=True).raw

# conversión de la imagen a objeto PIL
imagen = Image.open(fp=respuesta)

# fuente del texto OpenSans y tamaño de letra 100
fnt = ImageFont.truetype(font='../../../files/fonts/OpenSans-Light.ttf', size=100)

# inicio del dibujo sobre la imagen
dibujo = ImageDraw.Draw(im=imagen)

# texto en posición (50, 100) con la fuente elegida y en blanco
dibujo.text(xy=(50, 100), text='Kanagawa', font=fnt, fill=(255, 255, 255))

# texto en posición (200, 230) con la fuente elegida y en blanco
dibujo.text(xy=(200, 230), text='Wave', font=fnt, fill=(255, 255, 255))

imagen

# texto multilinea

# petición de la imagen
respuesta = req.get(url=url_img, stream=True).raw

# conversión de la imagen a objeto PIL
imagen = Image.open(fp=respuesta)

# fuente del texto OpenSans y tamaño de letra 70
fnt = ImageFont.truetype(font='../../../files/fonts/OpenSans-Light.ttf', size=70)

# inicio del dibujo sobre la imagen
dibujo = ImageDraw.Draw(im=imagen)

# texto multilínea(\n) en posición (50, 100) con la fuente elegida y en blanco
dibujo.multiline_text(xy=(50, 100), text='Kanagawa Wave\nHokusai\nKatsushika', font=fnt, fill=(255, 255, 255))


imagen

# texto transparente (marca de agua)


# petición de la imagen
respuesta = req.get(url=url_img, stream=True).raw

# conversión de la imagen a objeto PIL en formato RGBA
imagen = Image.open(fp=respuesta).convert('RGBA')

# fuente del texto OpenSans y tamaño de letra 40
fnt = ImageFont.truetype(font='../../../files/fonts/OpenSans-Light.ttf', size=40)

# imagen nueva para texto en formato rgba
txt = Image.new('RGBA', imagen.size, (255,255,255,0))

# inicio del dibujo sobre la imagen
dibujo = ImageDraw.Draw(txt)    

# texto en posición (400, 100) con la fuente elegida y negro transparente
dibujo.text((400, 100), 'Kanagawa', fill=(0, 0, 0, 40), font=fnt)

# combinación de la imagen con la transparencia
combinado = Image.alpha_composite(imagen, txt)    

combinado