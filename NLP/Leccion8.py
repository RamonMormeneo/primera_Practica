#traductor
from transformers import pipeline

tarea = 'translation'

modelo = 'Helsinki-NLP/opus-mt-en-es'
traductor = pipeline(task=tarea, model=modelo)


frase = '''Retrieval Augmented Generation (RAG) is a pattern that works with pretrained 
            Large Language Models (LLM) and your own data to generate responses.'''

traductor(frase)

print (traductor(frase)[0]['translation_text'])

#con Token

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizador = AutoTokenizer.from_pretrained(modelo)

vector = tokenizador(frase, return_tensors='pt')

modelo_traductor = AutoModelForSeq2SeqLM.from_pretrained(modelo)

respuesta = modelo_traductor.generate(**vector)

print (tokenizador.batch_decode(respuesta, skip_special_tokens=True)[0])

#resumen funcional

def traductor(frase: str, modelo: str) -> str:
    
    """
    Función para traducir un texto.
    
    Params:
    + frase: string. Frase o texto a ser traducido.
    + modelo: string. Modelo de transformers para traducir.
    
    Return:
    String. Frase o texto traducido.
    """
    
    # con este objeto vectorizamos las palabras
    tokenizador = AutoTokenizer.from_pretrained(modelo)
    
    
    # creación del vector
    vector = tokenizador(frase, return_tensors='pt')
    
    
    # inicializacion del modelo traductor
    modelo_traductor = AutoModelForSeq2SeqLM.from_pretrained(modelo)
    
    # tensor de respuesta del modelo
    respuesta = modelo_traductor.generate(**vector)
    
    # respuesta del modelo
    respuesta = tokenizador.batch_decode(respuesta, skip_special_tokens=True)[0]
    
    return respuesta

traductor(frase, modelo)