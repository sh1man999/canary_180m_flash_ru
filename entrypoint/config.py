import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
print(BASE_DIR)

HF_TOKEN = os.getenv("HF_TOKEN")