#generador de texto

from transformers import pipeline
tarea = 'text-generation'


modelo = 'openai-community/gpt2'

pipe_texto = pipeline(task=tarea, model=modelo)

frase = 'Hi, create a text about Rome'

print (pipe_texto(frase)[0]['generated_text'].split('\n'))

#ahora con ORca

modelo = 'M4-ai/Orca-2.0-Tau-1.8B'
pipe_texto = pipeline(task=tarea, model=modelo)

print(pipe_texto(frase))

#uso de chatbot

#prompt
mensajes = [{'role': 'system', 'content': 'You are un friendly chatbot who always responds in math way'},
            {'role': 'user', 'content': '2+1?'}
            ]

prompt = pipe_texto.tokenizer.apply_chat_template(conversation=mensajes,
                                                 tokenize=False,
                                                 add_generation_prompt=True
                                                )

#respuesta

respuesta = pipe_texto(text_inputs=prompt,
                       max_new_tokens=256,
                       do_sample=True,
                       temperature=0.2,
                       top_k=50,
                       top_p=0.95
                      )

print (respuesta[0]['generated_text'].split('assistant\n')[1])

#hacerlo funcional
from dotenv import load_dotenv      # carga variables de entorno 
import os                           # libreria de sistema
from transformers import pipeline   # importamos desde la librería transformers el pipeline

load_dotenv()


# importamos el token desde el archivo .env
HUGGINGFACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')



# definimos la tarea y el modelo 
TAREA = 'text-generation'  
MODELO = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'


# iniciamos el modelo de texto
PIPELINE = pipeline(task=TAREA, model=MODELO, token=HUGGINGFACE_TOKEN)



# iniciamos la variable prompt con el mensaje del sistema
PROMPT = '<|im_start|>system\nYou are a friendly chatbot who always responds in math way.<|im_end|>\n'

def update_context(contexto: str, user: bool = True) -> None:
    
    """
    Función para actualizar la conversación.
    
    Params:
    + contexto: string de la pregunta del usuario o de la respuesta del modelo.
    + user: booleano que nos indica si el usuario o el modelo que habla
    
    Return:
    No devueve nada, solo actualiza la variable global PROMPT
    """
    
    global PROMPT
    
    if user:
        PROMPT += f'\n<|im_start|>user\n{contexto}<|im_end|>\n<|im_start|>assistant\n'
        
    else:
        PROMPT += f'{contexto}<|im_end|>\n'
        
    return None

def chatbot(pregunta: str) -> str:
    
    """
    Función para llamar al modelo, recibe la pregunta y actualiza la conversación.
    
    Params:
    + pregunta: string con la pregunta del usuario
    
    Return:
    String con la respuesta del modelo
    """
    
    global PIPELINE, PROMPT
    
    
    # prompt de la conversacion 
    update_context(pregunta, user=True)
    
    
    # generamos la respuesta del chat
    respuesta = PIPELINE(text_inputs=PROMPT,
                         max_new_tokens=256, 
                         do_sample=True, 
                         temperature=0.2, 
                         top_k=50, 
                         top_p=0.95)
    
    
    # formato string de la respuesta
    respuesta = respuesta[0]['generated_text'].split('assistant\n')[-1].strip()
    
    
    # actualiza la conversacion con la respuesta
    update_context(respuesta, user=False)
    
    
    return respuesta

chatbot('2+1?')

#generacion text to text con Alpaca

tarea = 'text2text-generation'

modelo = 'declare-lab/flan-alpaca-large'

pipe_texto = pipeline(task=tarea, model=modelo, max_new_tokens=256)

print (pipe_texto('cuanto suma 2 y 2'))

print (pipe_texto('''

            responde la pregunta basandote en el siguiente contexto:
            
            contexto: Capital de Italia es Roma
            
            pregunta: ¿Cual es la capital?
            
           '''))

#generacion de codigo Gemma 2b

tarea = 'text2text-generation'

modelo = 'suriya7/Gemma-2B-Finetuned-Python-Model'

pipe_code = pipeline(task=tarea, model=modelo, max_new_tokens=512)

prompt = '''
<start_of_turn>user based on given instruction create a solution\n
here are the instruction filter pandas dataframe
<end_of_turn>\n<start_of_turn>model
'''
respuesta = pipe_code(prompt)

print (respuesta[0]['generated_text'].split('\n'))

#una forma de ver mejor la respuesta es Markdown

from IPython.display import Markdown

display(Markdown(respuesta[0]['generated_text']))