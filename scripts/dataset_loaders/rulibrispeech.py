import json
import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

from config import BASE_DIR  # Импортируем базовую директорию из конфига
from utils import normalize_text

# Создаём директории для файлов
os.makedirs(os.path.join(BASE_DIR, 'datasets/rulibri'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'datasets/rulibri/audio/train'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'datasets/rulibri/audio/validate'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'datasets/rulibri/audio/test'), exist_ok=True)

# Загружаем датасет
rulibri = load_dataset('bond005/rulibrispeech')


# Функция для создания манифеста и сохранения аудио
def create_manifest(split, output_path, audio_folder):
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(tqdm(rulibri[split], desc=f"Обработка {split}")):
            # Создаем имя файла с индексом для уникальности
            orig_filename = os.path.basename(item["audio"]["path"])
            filename = f"{split}_{idx}_{orig_filename}"

            # Относительный и абсолютный пути
            rel_path = f"./datasets/rulibri/audio/{audio_folder}/{filename}"
            abs_path = os.path.join(BASE_DIR, f'datasets/rulibri/audio/{audio_folder}/{filename}')

            # Сохраняем аудио на диск
            sf.write(
                abs_path,  # Используем абсолютный путь для записи
                item["audio"]["array"],
                item["audio"]["sampling_rate"]
            )

            # Создаём запись для манифеста с относительным путем
            entry = {
                "audio_filepath": rel_path,  # Относительный путь
                "duration": len(item["audio"]["array"]) / item["audio"]["sampling_rate"],
                "text": normalize_text(item["transcription"]),
                "source_lang": "ru",
                "target_lang": "ru",
                "taskname": "asr",
                "pnc": "yes"
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# Создаём манифесты и сохраняем аудио
create_manifest('train',
                os.path.join(BASE_DIR, 'datasets/rulibri/train_manifest.json'),
                'train')

create_manifest('validation',
                os.path.join(BASE_DIR, 'datasets/rulibri/validate_manifest.json'),
                'validate')

create_manifest('test',
                os.path.join(BASE_DIR, 'datasets/rulibri/test_manifest.json'),
                'test')

print(f"\nГотово! Манифесты сохранены в директории {os.path.join(BASE_DIR, 'datasets/rulibri/')}")