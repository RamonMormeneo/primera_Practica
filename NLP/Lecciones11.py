#relleno de espacios

from transformers import pipeline

tarea = 'fill-mask'

modelo = 'google-bert/bert-base-multilingual-cased'
fill_pipe = pipeline(task=tarea, model=modelo, device='cpu')

frase = 'La capital de España es [MASK].'

#Nos saldra un dic ordenado de mas probabilidad a menos

fill_pipe(frase)

#token

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizador = AutoTokenizer.from_pretrained(modelo)

vector = tokenizador(frase, return_tensors='pt')

modelo_rellenado = AutoModelForMaskedLM.from_pretrained(modelo)

respuesta = modelo_rellenado(**vector)

logits = respuesta.logits

vector.input_ids == tokenizador.mask_token_id

mask_index = (vector.input_ids == tokenizador.mask_token_id)[0].nonzero()[0]

logits[0, mask_index].argmax(axis=-1)

resp = tokenizador.decode(logits[0, mask_index].argmax(axis=-1))

frase.replace('[MASK]', resp)

print(frase)