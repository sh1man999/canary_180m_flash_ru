import json
import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

# Загружаем датасет
rulibri = load_dataset('bond005/rulibrispeech')

# Создаём директории для файлов и манифестов
os.makedirs('audio_files', exist_ok=True)
os.makedirs('manifests', exist_ok=True)


# Функция для создания манифеста и сохранения аудио
def create_manifest(split, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(rulibri[split], desc=f"Обработка {split}"):
            # Извлекаем имя файла из пути
            filename = os.path.basename(item["audio"]["path"])

            # Формируем локальный путь для сохранения
            local_path = os.path.join("audio_files", filename)

            # Сохраняем аудио на диск
            sf.write(
                local_path,
                item["audio"]["array"],
                item["audio"]["sampling_rate"]
            )

            # Создаём запись для манифеста
            entry = {
                "audio_filepath": os.path.abspath(local_path),  # Абсолютный путь
                "duration": len(item["audio"]["array"]) / item["audio"]["sampling_rate"],
                "text": item["transcription"].replace("(", "").replace(")", "").replace(" .", ".").replace("\"","").strip(),
                "source_lang": "ru",
                "target_lang": "ru",
                "taskname": "asr",
                "pnc": "yes"
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# Создаём манифесты и сохраняем аудио (тут можно выбрать нужные сплиты)
create_manifest('train', 'manifests/train_manifest.json')
create_manifest('validation', 'manifests/val_manifest.json')
create_manifest('test', 'manifests/test_manifest.json')