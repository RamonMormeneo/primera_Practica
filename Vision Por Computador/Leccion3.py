#Open CV

# importamos OpenCV

import cv2

ruta = 'ficehros/tokyoGettyImages-1031467664.webp'

# carga de la imagen

imagen = cv2.imread(filename=ruta)
# tipo de dato de la imagen

print(type(imagen))

# dimensiones de la imagen

print(imagen.shape)

# importamos PIL

from PIL import Image

# convertimos el array de numpy a imagen

Image.fromarray(obj=imagen)

# importamos numpy

import numpy as np

# filtro del array, ponemos 0 si el pixel tiene un valor mayor de 180

filtro = np.where(imagen < 180, 0, imagen)

# visualizamos la imagen filtrada

Image.fromarray(obj=filtro)

# guardado

ruta_guardado = 'ficehros/tokyoGettyImages-1031467664.webp'

cv2.imwrite(filename=ruta_guardado, img=filtro)

#Transformaciones

# redimensionado

dimensiones = (100 ,100)

nueva_imagen = cv2.resize(src=imagen, dsize=dimensiones)

nueva_imagen.shape

# visualizacion de la imagen

Image.fromarray(obj=nueva_imagen)

# definimos el ancho y alto de la imagen

altura, ancho = imagen.shape[:2]

altura, ancho

# definimos el punto de rotación, el centro de la imagen por ejemplo

punto_rotacion = (ancho//2, altura//2)

punto_rotacion

# definimos el angulo de rotación en grados, antihorario

angulo = 45

# ahora construimos la matriz de rotación

matriz_rotacion = cv2.getRotationMatrix2D(center=punto_rotacion, angle=angulo, scale=1.0)

# rotación de la imagen

rotacion = cv2.warpAffine(src=imagen, M=matriz_rotacion, dsize=(ancho, altura))

# visualización

Image.fromarray(rotacion)

# transposición

transpuesta = cv2.flip(src=imagen, flipCode=-1)

# visualización

Image.fromarray(obj=transpuesta)

# recorte

recorte = imagen[100:200, 0:100]

recorte.shape

# visualización

Image.fromarray(obj=recorte)

#FILTROS

# desenfoque

desenfoque = cv2.blur(src=imagen, ksize=(10, 10))

# visualización

Image.fromarray(obj=desenfoque)

# conversión a escala de grises

gris = cv2.cvtColor(src=imagen, code=cv2.COLOR_BGR2GRAY)

Image.fromarray(gris)

# umbral en valor 150, si está por debajo 0, si está por encima 255

binario = cv2.threshold(src=gris, thresh=150, maxval=255, type=cv2.THRESH_BINARY)

# visualización 

Image.fromarray(binario[1])

# en negativo

binario_neg = cv2.threshold(src=gris, thresh=150, maxval=255, type=cv2.THRESH_BINARY_INV)

Image.fromarray(binario_neg[1])

#MASCARAS

# pintamos el círculo

circulo = cv2.circle(img=imagen.copy(), center=(180, 100), radius=100, color=255, thickness=-1)

# visualización

Image.fromarray(obj=circulo)

rectangulo = cv2.rectangle(img=imagen.copy(), pt1=(40,70), pt2=(300,300), color=255, thickness=-1)

Image.fromarray(obj=rectangulo)

# imagen en blanco con las dimensiones de la imagen original
en_blanco = np.zeros(shape=imagen.shape[:2], dtype='uint8')


# pintamos el círculo
circulo = cv2.circle(img=en_blanco.copy(), center=(180, 100), radius=100, color=255, thickness=-1)


# creamos la nueva imagen enmascarada
mascara = cv2.bitwise_and(src1=imagen, src2=imagen, mask=circulo)

Image.fromarray(obj=mascara)

# imagen en blanco con las dimensiones de la imagen original salvo los canales
en_blanco = np.zeros(shape=imagen.shape[:2], dtype='uint8')


# pintamos el rectángulo
rectangulo = cv2.rectangle(img=en_blanco.copy(), pt1=(40,70), pt2=(300,300), color=255, thickness=-1)


# creamos la nueva imagen enmascarada
mascara = cv2.bitwise_and(src1=imagen, src2=imagen, mask=rectangulo)

Image.fromarray(obj=mascara)

#Reconocimiento de cointronos

# conversión a escala de grises

gris = cv2.cvtColor(src=imagen, code=cv2.COLOR_BGR2GRAY)

# desenfoque de la imagen en escala de grises

desenfoque = cv2.blur(src=gris, ksize=(3, 3))

# encontramos los contornos dentro de la imagen

contorneada = cv2.Canny(image=desenfoque, threshold1=0, threshold2=255)

# visualización

Image.fromarray(obj=contorneada)

# imagen en blanco con las dimensiones de la imagen original

en_blanco = np.zeros(shape=imagen.shape, dtype='uint8')

# extracción de contornos y jerarquía

contornos, jerarquia = cv2.findContours(image=contorneada, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

# dibujamos los contornos en la imagen en blanco

n_contorneada = cv2.drawContours(image=en_blanco, 
                                 contours=contornos, 
                                 contourIdx=-1, 
                                 color=(0,0,255), 
                                 thickness=1)

Image.fromarray(obj=n_contorneada)

#RECONOCIMIENTO DE CARARAS

# ruta que contiene los modelos clasificadores

cv2.data.haarcascades

# lista de clasificadores que tiene opencv

import os

os.listdir(cv2.data.haarcascades)

def detectar(ruta_imagen: str, modelo: str, escala: float = 1.3, vecinos: int = 1) -> None:
    
    """
    Esta función recibe la ruta de la imagen, el tipo de modelo, el factor de escala 
    y mínimo de vecinos para enseñar la imagen clasificada.
    
    Params:
    ruta_imagen: string, ruta a la imagen que queremos clasificar
    modelo: string, tipo de modelo haar que vamos a usar
    escala: float, factor de escala
    vecinos: int, número de vecinos
    
    Returns
    None. No devuelve nada, solo nos enseña la imagen clasificada.
    
    """
    
    imagen = cv2.imread(filename=ruta_imagen)


    gris = cv2.cvtColor(src=imagen, code=cv2.COLOR_BGR2GRAY)


    modelo = cv2.CascadeClassifier(filename=cv2.data.haarcascades + modelo)


    rectangulos = modelo.detectMultiScale(image=gris, scaleFactor=escala, minNeighbors=vecinos)


    for x,y,w,h in rectangulos:

        cv2.rectangle(img=imagen, pt1=(x,y), pt2=(x+w,y+h), color=(0,255,0), thickness=2)
        
        
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)


    display(Image.fromarray(imagen))

detectar(ruta_imagen='ficehros/Grupo_de_amigos_636032262.webp', modelo='haarcascade_frontalface_alt.xml')

