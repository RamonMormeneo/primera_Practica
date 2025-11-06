from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#silencio de warnings
from transformers import logging

logging.set_verbosity_error()

# Selecionar el modelo y la tarea que le queremos dar
tarea = 'text-classification'

modelo = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'

clasificador = pipeline(task=tarea, model=modelo)

clasificador('me gusta aprender')

lista = ['me gusta aprender', 'hola, buenos dias', 'no me gustas']

clasificador(lista)

#Usar el modelo directamente

tokenizador = AutoTokenizer.from_pretrained(modelo)

# Aqui has varios elementos
# vocab_size: Se refiere al tamaño del vocabulario del que maneja, define el número de diferentes tokens que se pueden representar.
# model_max_length: Número máximo de tokens de salida del tokenizador.
# special_tokens: Estos son tokens especiales utilizados en la arquitectura de los modelos Transformer, especialmente en modelos como BERT y sus variantes, como la versión DistilBert que estamos usando. Aquí está el significado de cada uno:
#   [PAD]: Este token se utiliza para hacer que todas las secuencias de entrada tengan la misma longitud, rellenando las secuencias más cortas con este token hasta alcanzar la longitud máxima.
#   [UNK]: Representa palabras desconocidas o fuera del vocabulario durante el entrenamiento o la inferencia. Se utiliza cuando el modelo encuentra una palabra que no está en su vocabulario predefinido.
#   [CLS]: Este token se agrega al principio de cada secuencia de entrada. Se utiliza en tareas de clasificación de texto para representar la clase de toda la secuencia.
#   [SEP]: Se utiliza para separar dos oraciones o fragmentos de texto en una sola secuencia de entrada. También se agrega al final de cada secuencia de entrada.
#   [MASK]: Este token se utiliza en tareas de llenado de espacios ([cloze tasks](https://es.wikipedia.org/wiki/Prueba_cloze)) durante el entrenamiento del modelo. Se sustituye aleatoriamente por una palabra en la secuencia de entrada, y el modelo debe predecir la palabra original.

frase = 'me gusta el aprendizaje'

vector = tokenizador(frase, return_tensors='pt')

vector

clasificador = AutoModelForSequenceClassification.from_pretrained(modelo)

#IMPORTANTE PASAR EL VECTOR COMO KEYWORD

clasificador(**vector).logits

respuesta = clasificador(**vector).logits.argmax()

respuesta = respuesta.item()  

#Funciones Custom

from transformers import pipeline
from typing import Union

def hugging_pipeline(frase: Union[str,list], tarea: str, modelo: str) -> list:
    
    
    clasificador = pipeline(task=tarea, model=modelo)
    
    respuesta = clasificador(frase)
    
    return respuesta

frase = 'me gusta aprender'

tarea = 'text-classification'

modelo = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'

hugging_pipeline(frase, tarea, modelo)

hugging_pipeline(['hola que tal', 'no me agradas'], tarea, modelo)

#Ahora usando el modelo Roberta

modelo = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

hugging_pipeline(frase, tarea, modelo)
hugging_pipeline(['hola que tal', 'no me agradas'], tarea, modelo)
#son normalmente mejores en ingles
hugging_pipeline('you are a dog', tarea, modelo)



#respuesta custom

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def hugging_model(frase: str, modelo: str) -> dict:
    
    
    tokenizador = AutoTokenizer.from_pretrained(modelo)
    
    vector = tokenizador(frase, return_tensors='pt')
    
    
    
    clasificador = AutoModelForSequenceClassification.from_pretrained(modelo)
    
    
    tensor = clasificador(**vector).logits
    
    respuesta = tensor.argmax().item()
    
    
    return {'frase': frase,
            'modelo': modelo,
            'tensor': tensor,
            'n_respuesta': respuesta,
            'respuesta': 'positivo' if respuesta==1 else 'negativo'
           }

frase = 'no me agradas'

modelo = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'

hugging_model(frase, modelo)