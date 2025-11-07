#Responder Preguntas

from transformers import logging
from transformers import pipeline

logging.set_verbosity_error()

tarea = 'question-answering'

modelo = 'mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'

qa_pipe = pipeline(task=tarea, model=modelo)

contexto = 'El español es el segundo idioma más hablado del mundo con más de 442 millones de hablantes.'

pregunta = '¿Cuantas personas hablan español?'

prompt = {'context': contexto,
          'question': pregunta}

# 2 formas de usar el pipeline

qa_pipe(prompt)

qa_pipe(context=contexto, question=pregunta)

#modelo QA

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizador = AutoTokenizer.from_pretrained(modelo)

vector = tokenizador(pregunta, contexto, return_tensors='pt')

tokens = vector.tokens()

modelo_qa = AutoModelForQuestionAnswering.from_pretrained(modelo)

resultado = modelo_qa(**vector)

#sacar el inicio de la respuesta
indice_inicial = resultado.start_logits.argmax()

#sacar el fianl de la respuesta
indice_final = resultado.end_logits.argmax()

#esta salida necesita que la transformemos
print(tokens[indice_inicial: indice_final])

#pasamos a ver otra forma de hacer esto y que este bien escrito

vector.input_ids[0]

print (tokenizador.decode(vector.input_ids[0]))

tensor_respuesta = vector.input_ids[0, indice_inicial: indice_final]

print (tokenizador.decode(tensor_respuesta))

#ejemplo de uso 

def qa(pregunta: str, contexto: str, modelo: str) -> str:
    
    
    tokenizador = AutoTokenizer.from_pretrained(modelo)
    
    vector = tokenizador(pregunta, contexto, return_tensors='pt')
    
    modelo_qa = AutoModelForQuestionAnswering.from_pretrained(modelo)
    
    resultado = modelo_qa(**vector)
    
    indice_inicial = resultado.start_logits.argmax()
    
    indice_final = resultado.end_logits.argmax()
    
    tensor_respuesta = vector.input_ids[0, indice_inicial : indice_final]
    
    respuesta = tokenizador.decode(tensor_respuesta, skip_special_tokens=True)
    
    return respuesta

pregunta = '¿Cuantas personas hablan español?'

contexto = 'El español es el segundo idioma más hablado del mundo con más de 442 millones de hablantes.'

modelo = 'mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'

print (qa(pregunta, contexto, modelo))