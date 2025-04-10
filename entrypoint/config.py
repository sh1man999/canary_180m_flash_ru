import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

HF_TOKEN = os.getenv("HF_TOKEN")