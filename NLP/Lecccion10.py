#extraccion de caracteristicas y similitudes

from transformers import pipeline

tarea = 'feature-extraction'

modelo = 'Supabase/gte-small'

gte_pipe = pipeline(task=tarea, model=modelo)

frase = 'hola, buenos dias'

gte_pipe(frase, return_tensors=True).shape

modelo = 'facebook/bart-base'

bart_pipe = pipeline(task=tarea, model=modelo)

frase = 'hola, buenos dias'

bart_pipe(frase, return_tensors=True).shape

frase = 'hola, buenos dias, ¿como estas?. Yo bien ¿y tu?. Aqui estamos'

bart_pipe(frase, return_tensors=True).shape

#tokenizador

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizador = AutoTokenizer.from_pretrained(modelo)

frase = 'hola, buenos dias'

respuesta = tokenizador(frase, return_tensors='pt')

respuesta['input_ids'].shape

frase = 'hola, buenos dias, ¿como estas?. Yo bien ¿y tu?. Aqui estamos'

respuesta = tokenizador(frase, return_tensors='pt')

respuesta['input_ids'].shape

tokenizador.batch_decode(respuesta['input_ids'])

print(respuesta.tokens())

#transformador de frases

modelo = 'sentence-transformers/all-roberta-large-v1'

from sentence_transformers import SentenceTransformer

transformador = SentenceTransformer(modelo)

frase = 'hola, buenos dias'

print(transformador.encode(frase).shape)

frase = 'hola, buenos dias, ¿como estas?. Yo bien ¿y tu?. Aqui estamos'

print(transformador.encode(frase).shape)

frases = ['hola Alegre', 'estamos en clase de IA', 'estamos aprendiendo embeddings']

transformador.encode(frases).shape

#similitu de oraciones

from sklearn.metrics.pairwise import cosine_similarity

frases = ['hola alegre', 'hola galan']

vectores = transformador.encode(frases)

cosine_similarity([vectores[0]], [vectores[1]])