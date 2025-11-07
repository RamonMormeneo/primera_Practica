#respuestas a prrguntas con tablas como contexto
import pandas as pd

tabla = pd.read_csv('ficheros/asturias_viajeros_por_franja.csv', sep=';')

tabla.head()
tabla.info(memory_usage='deep')

tabla_estacion = tabla.groupby('NOMBRE_ESTACION').agg({'VIAJEROS_SUBIDOS': 'sum',
                                                       'VIAJEROS_BAJADOS': 'sum'}).reset_index()

tabla_estacion.info()

tabla_hora = tabla.groupby('TRAMO_HORARIO').agg({'VIAJEROS_SUBIDOS': 'sum',
                                                       'VIAJEROS_BAJADOS': 'sum'}).reset_index()

tabla_hora.info()

tabla_hora.head().to_dict(orient='list')

#pipline

from transformers import pipeline

tarea = 'table-question-answering'

modelo = 'microsoft/tapex-large-finetuned-wtq'

tqa_pipe = pipeline(task=tarea, model=modelo)

pregunta = 'quitando Oviedo, en que estacion se suben más viajeros'

tqa_pipe(query=pregunta, table=tabla_estacion.to_dict(orient='list'))

prompt = {'query': pregunta, 'table': tabla_estacion.to_dict(orient='list')}
print(tqa_pipe(prompt))

print (tabla_estacion.sort_values(by='VIAJEROS_SUBIDOS', ascending=False).head())

#tokenizador

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizador = AutoTokenizer.from_pretrained(modelo)

vector = tokenizador(table=tabla_estacion, query=pregunta, return_tensors='pt')

vector['input_ids'].shape

modelo_tqa = AutoModelForSeq2SeqLM.from_pretrained(modelo)

resultado = modelo_tqa.generate(**vector)

print(tokenizador.batch_decode(resultado, skip_special_tokens=True)[0])

# todo el codigo en 1 sitio

def consulta(pregunta):
    
    
    tabla = pd.read_csv('ficheros/asturias_viajeros_por_franja.csv', sep=';')


    tabla_estacion = tabla.groupby('NOMBRE_ESTACION').agg({'VIAJEROS_SUBIDOS': 'sum',
                                                           'VIAJEROS_BAJADOS': 'sum'}).reset_index()



    tokenizador = AutoTokenizer.from_pretrained(modelo)


    modelo_tqa = AutoModelForSeq2SeqLM.from_pretrained(modelo)


    vector = tokenizador(table=tabla_estacion, query=pregunta, return_tensors='pt')


    resultado = modelo_tqa.generate(**vector)

    return tokenizador.batch_decode(resultado, skip_special_tokens=True)[0]

print (consulta('quitando Oviedo, en que estacion se suben más viajeros'))