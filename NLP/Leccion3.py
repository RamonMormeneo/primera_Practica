from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv('HUGGING_FACE_TOKEN')

print(token)