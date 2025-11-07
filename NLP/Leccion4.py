#Classificacion de Tokens

from transformers import pipeline

tarea = 'ner'

modelo = 'Babelscape/wikineural-multilingual-ner'

#aggrefatios strategy es para agrupar los B-X y los I-X pero tambien puede ser, first, average y max
ner_pipe = pipeline(task=tarea, model=modelo, aggregation_strategy='simple')

frase = 'Mi nombre es Juan, vivo en Madrid y trabajo en BBVA'

#Te devuelve unos diccionarios donde tienen claves que indican cosas diferentes como word, la palabra reconocida, entity, score, index, start y end

ner_pipe(frase)

from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizador = AutoTokenizer.from_pretrained(modelo)

vector = tokenizador(frase, return_tensors='pt')

#podemos ver que nos ha separado el texto por palabras
tokens = vector.tokens()

#Modelo Classificador
modelo_ner = AutoModelForTokenClassification.from_pretrained(modelo)
tensor = modelo_ner(**vector).logits

etiquetas = modelo_ner.config.id2label
#veremos que sale 1,15,9 1 capa con 15 filas y 9 columnas y con esto podemos saber diferente informacion
tensor.shape

indices = [int(e.argmax()) for e in tensor[0]]

entidades = [etiquetas[e] for e in indices]

#juntar en dic
dict(zip(tokens, entidades))

#combinar

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification


tarea = 'ner'

modelo = 'Babelscape/wikineural-multilingual-ner'


tokenizador = AutoTokenizer.from_pretrained(modelo)
modelo_ner = AutoModelForTokenClassification.from_pretrained(modelo)


ner_pipe = pipeline(task=tarea, model=modelo_ner, tokenizer=tokenizador, aggregation_strategy='simple')



frase = '''
        Mi nombre es Juan, vivo en Madrid,
        aunque a veces voy a Barcelona.
        Mi perro se llama Nerón.
        Trabajo en BBVA, pero mi ultimo puesto fue en Vodafone.
        '''



resultado = ner_pipe(frase)

resultado

#como visualizar mejor las entidades usando ipymarkup

rangos_ents = [(e['start'], e['end'], e['entity_group']) for e in resultado]

#from ipymarkup import show_span_ascii_markup

#show_span_ascii_markup(frase, rangos_ents)

from ipymarkup.palette import palette, BLUE, RED, GREEN, ORANGE

from ipymarkup import show_span_line_markup

show_span_line_markup(frase, rangos_ents)

show_span_line_markup(frase, rangos_ents, palette=palette(BLUE))

from ipymarkup import show_span_box_markup

show_span_box_markup(frase, rangos_ents, palette=palette(PER=ORANGE, ORG=BLUE, LOC=RED))

texto = show_span_box_markup(frase, rangos_ents, palette=palette(PER=ORANGE, ORG=BLUE, LOC=RED))

from ipymarkup import format_span_box_markup

html = list(format_span_box_markup(frase, rangos_ents))

print(html)

from ipymarkup import format_span_box_markup
html = list(format_span_box_markup(frase, rangos_ents))
print(html)
