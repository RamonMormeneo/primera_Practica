#tunning i dataset

from datasets import load_dataset

glue_dataset = load_dataset('glue', 'mrpc')

glue_dataset['train'][0]

glue_dataset['train'].features

#analisis de sentimiento X

tweet_dataset = load_dataset('mteb/tweet_sentiment_extraction')

tweet_dataset

tweet_dataset['train'][0]

tweet_dataset['train'].features

#fine tunning GLUE del BERT

from transformers import AutoTokenizer, AutoModelForSequenceClassification

modelo = 'bert-base-uncased'

tokenizador = AutoTokenizer.from_pretrained(modelo)
modelo_bert = AutoModelForSequenceClassification.from_pretrained(modelo, num_labels=2)

token_dataset = glue_dataset.map(lambda x: tokenizador(x['sentence1'],
                                                       x['sentence2'],
                                                       truncation=True
                                                      ),
                                
                                batched=True)

print(token_dataset['train'][0]['input_ids'])
#data collector
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizador)

#trainer

from transformers import Trainer, TrainingArguments

arg_training = TrainingArguments(output_dir='training/glue-trainer')

print(arg_training)

entrenador = Trainer(model=modelo_bert,
                     tokenizer=tokenizador,
                     train_dataset=token_dataset['train'],
                     eval_dataset=token_dataset['validation'],
                     args=arg_training,
                     data_collator=data_collator
                    )

entrenador.train()

#Prediccion y evaluacion

predicciones = entrenador.predict(token_dataset['test'])

predicciones.predictions[0]
# etiquetas

etiquetas = predicciones.label_ids

etiquetas[0]

import numpy as np

predicciones = np.argmax(predicciones.predictions, axis=-1)

print(predicciones.shape, etiquetas.shape)

# con lo que hemoos sacado, caluculamos el acierto, accuracy y f1 score

entrenador.evaluate()

# importamos las métricas de evaluación

from sklearn.metrics import accuracy_score, f1_score

# calculamos las métricas

{'accuracy':accuracy_score(predicciones, etiquetas), 
 'f1':f1_score(predicciones, etiquetas)}

def evaluacion(modelo_preds):
    
    """
    Función para obtener la métricas de evaluación.
    
    Params:
    + modelo_preds: transformers.trainer_utils.PredictionOutput, predicciones del modelo y etiquetas.
    
    Return:
    dict: diccionario con keys accuracy y f1-score y sus valores respectivos.
    """
    
    preds, etiquetas = modelo_preds
    
    preds = np.argmax(preds, axis=-1)
        
    return {'accuracy': accuracy_score(preds, etiquetas), 
            'f1': f1_score(preds, etiquetas)}



# argumento de entrenamiento

args_entrenamiento = TrainingArguments(output_dir='training/glue-trainer', 
                                       evaluation_strategy='steps',
                                       logging_steps=100,
                                      )

# entrenador con función de métricas

entrenador = Trainer(model=modelo_bert,
                     args=args_entrenamiento,
                     train_dataset=token_dataset['train'],
                     eval_dataset=token_dataset['validation'],
                     data_collator=data_collator,
                     tokenizer=tokenizador,
                     compute_metrics=evaluacion
                    )

# entrenamiento

entrenador.train()

# evaluación directa desde el entrenador, ahora con acierto y f1

entrenador.evaluate()

#Como Guardar el modelo

# guardado en la ruta dada

entrenador.save_model('training/glue_bert')

#Fine tunning en analisis de sentimiento
# dataset

tweet_dataset = load_dataset('mteb/tweet_sentiment_extraction')

# definimos el modelo 

modelo = 'bert-base-uncased'

# con este objeto se vectorizan las palabras

tokenizador = AutoTokenizer.from_pretrained(modelo)

# iniciamos el modelo BERT

modelo_bert = AutoModelForSequenceClassification.from_pretrained(modelo, num_labels=3)

# tokenizar el dataset

token_dataset = tweet_dataset.map(lambda x: tokenizador(x['text'], 
                                                        truncation=True), 
                                  batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizador)

# argumentos de entrenamiento

args_entrenamiento = TrainingArguments(output_dir='training/sentiment-trainer', 
                                       evaluation_strategy='steps',
                                       logging_steps=1000,
                                      )

# función de evaluación

def evaluacion(model_preds):
    
    """
    Función para obtener la métricas de evaluación.
    
    Params:
    + modelo_preds: transformers.trainer_utils.PredictionOutput, predicciones del modelo y etiquetas.
    
    Return:
    dict: diccionario con keys accuracy y f1-score y sus valores respectivos. 
    Se añade macro al f1-score porque no es un clasificador binario, sino que hay 3 clases.
    """
    
    preds, labels = model_preds
    
    preds = np.argmax(preds, axis=-1)
        
    return {'accuracy': accuracy_score(preds, labels), 
            'f1': f1_score(preds, labels, average='macro')}

# entrenador 

entrenador = Trainer(model=modelo_bert,
                     args=args_entrenamiento,
                     train_dataset=token_dataset['train'],
                     eval_dataset=token_dataset['test'],
                     data_collator=data_collator,
                     tokenizer=tokenizador,
                     compute_metrics=evaluacion
                    )

# entrenamiento

entrenador.train()
# evaluación directa desde el entrenador

entrenador.evaluate()

# evaluación manual

preds = entrenador.predict(token_dataset['test'])

etiquetas = preds.label_ids

preds = np.argmax(preds.predictions, axis=-1)

{'accuracy':accuracy_score(preds, etiquetas), 
 'f1':f1_score(preds, etiquetas, average='macro')}

# guardado del modelo

entrenador.save_model('training/sentiment_bert')
