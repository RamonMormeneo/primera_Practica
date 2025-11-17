#Audio a audio

# para quitar warnings 

from transformers import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore')

# librerías

import torchaudio
from speechbrain.inference.separation import SepformerSeparation 
import librosa
from IPython.display import Audio

# audio original

ruta = 'audio_cache/test_mixture_3spks.wav'

data, freq = librosa.load(ruta)

Audio(data=data, rate=freq)

# modelo separador

modelo = SepformerSeparation.from_hparams(source='speechbrain/sepformer-wsj03mix')

# separando fuentes

fuentes = modelo.separate_file(path='speechbrain/sepformer-wsj03mix/test_mixture_3spks.wav')

# 1ª fuente

Audio(data=fuentes[:, :, 0].detach().cpu(), rate=8000)

# 2ª fuente

Audio(data=fuentes[:, :, 1].detach().cpu(), rate=8000)

# 3ª fuente

Audio(data=fuentes[:, :, 2].detach().cpu(), rate=8000)

# librerías

import torch
import numpy as np
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan

# carga de audio original con voz masculina desde un dataset

dataset = load_dataset('hf-internal-testing/librispeech_asr_demo', 'clean', split='validation')


data = dataset[0]['audio']['array']

freq = dataset.features['audio'].sampling_rate


Audio(data=data, rate=freq)

# carga del modelo, el procesador y el vocoder


ruta = 'microsoft/speecht5_vc'


procesador = SpeechT5Processor.from_pretrained(ruta)

modelo = SpeechT5ForSpeechToSpeech.from_pretrained(ruta)

vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')  

# creación del tensor

tensor = procesador(audio=data, sampling_rate=freq, return_tensors='pt')

# embedding de audio femenino

embedding_data = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')

embedding = torch.tensor(embedding_data[7306]['xvector']).unsqueeze(0)

# generación de voz femenina

resultado = modelo.generate_speech(tensor['input_values'], embedding, vocoder=vocoder)

# escuchamos nuevo audio

Audio(data=resultado.numpy(), rate=16000)