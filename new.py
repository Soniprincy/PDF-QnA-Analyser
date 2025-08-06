import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

models = genai.list_models()
for model in models:
    print(model.name, model.supported_generation_methods)
