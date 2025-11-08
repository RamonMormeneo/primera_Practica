#classificacion 0 shot
from transformers import pipeline

tarea = 'zero-shot-classification'

modelo = 'facebook/bart-large-mnli'

zero_pipe = pipeline(task=tarea, model=modelo)

etiquetas = ['urgente', 'no urgente', 'reparacion', 'revision']

pregunta = 'Tengo un problema con mi computadora que necesita ser resuelto lo antes posible. De ello depende mi trabajo'

zero_pipe(sequences=pregunta, candidate_labels=etiquetas, multi_label=True)

prompt = {'sequences': pregunta, 'candidate_labels': etiquetas}

zero_pipe(**prompt)

respuesta = zero_pipe(**prompt)

print (respuesta['labels'][0])

prompt = {'sequences': 'Hola, mi coche funciona pero hace un ruido', 
          'candidate_labels': etiquetas}

respuesta = zero_pipe(**prompt)

print (respuesta['labels'][0])

prompt = {'sequences': 'Hola, mi telefono no suena', 
          'candidate_labels': etiquetas}

respuesta = zero_pipe(**prompt)

print (respuesta['labels'][0])

#ahora usando el modelo zero shot tokenizador

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizador = AutoTokenizer.from_pretrained(modelo)
vector = tokenizador.encode(pregunta, etiquetas[0], return_tensors='pt')

print (tokenizador.decode(vector[0]))

zero_modelo = AutoModelForSequenceClassification.from_pretrained(modelo)
respuesta = zero_modelo(vector)

print (respuesta.logits)

#resumen funcional

def zero_shot(pregunta: str, etiqueta: str, modelo: str) -> float:
    
    """
    Función para zero shot
    """
    
    
    # con este objeto vectorizamos las palabras
    tokenizador = AutoTokenizer.from_pretrained(modelo)
    
    
    # creación del vector
    vector = tokenizador.encode(pregunta, etiqueta, return_tensors='pt')
    
    
    # inicializacion del modelo Zero Shot
    zero_modelo = AutoModelForSequenceClassification.from_pretrained(modelo)
    
    
    # respuesta del modelo al pasarle el vector
    respuesta = zero_modelo(vector)
    
    
    # probabilidad de pertenencia
    probabilidad = respuesta.logits.softmax(dim=1)[0, 1].item()
    
    
    return probabilidad

pregunta = 'Tengo un problema con mi computadora que necesita ser resuelto lo antes posible. De ello depende mi trabajo'

etiqueta = 'revision'

modelo = 'facebook/bart-large-mnli'

zero_shot(pregunta, etiqueta, modelo)


for et in etiquetas:
    
    res = zero_shot(pregunta, et, modelo)
    
    print(et, res)