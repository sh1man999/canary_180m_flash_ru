import json
import os
import uuid
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

from config import BASE_DIR, HF_TOKEN
from utils import normalize_text

# Создаём необходимые директории
os.makedirs(os.path.join(BASE_DIR, 'datasets/common_voice'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'datasets/common_voice/audio/train'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'datasets/common_voice/audio/validate'), exist_ok=True)

# Открываем файлы манифестов
train_file = open(os.path.join(BASE_DIR, 'datasets/common_voice/train_manifest.json'), 'w', encoding='utf-8')
val_file = open(os.path.join(BASE_DIR, 'datasets/common_voice/validate_manifest.json'), 'w', encoding='utf-8')

# Статистика
stats = {'train': 0, 'val': 0, 'filtered_train': 0, 'filtered_val': 0}


# Функция обработки набора данных
def process_dataset(dataset_iter, output_file, audio_dir, counter_key, filtered_key):
    for idx, item in enumerate(tqdm(dataset_iter)):
        # Пропускаем запись, если есть отрицательные голоса
        if item.get("down_votes", 0) > 0:
            continue

        stats[filtered_key] += 1

        # Создаем имя файла
        filename = f"record_{idx}_{uuid.uuid4().hex[:8]}.wav"

        # Относительный и абсолютный пути
        audio_path = os.path.join(BASE_DIR, f'datasets/common_voice/audio/{audio_dir}/{filename}')
        rel_path = f"./datasets/common_voice/audio/{audio_dir}/{filename}"

        # Сохраняем аудио
        sf.write(
            audio_path,
            item["audio"]["array"],
            item["audio"]["sampling_rate"]
        )

        # Обработка текста
        text = item["sentence"]
        text = normalize_text(text)
        text = ' '.join(text.split())

        # Создаём запись для манифеста с относительным путем
        entry = {
            "audio_filepath": rel_path,  # Используем относительный путь
            "duration": len(item["audio"]["array"]) / 16000,
            "text": text.strip(),
            "source_lang": "ru",
            "target_lang": "ru",
            "taskname": "asr",
            "pnc": "yes"
        }

        # Записываем в манифест
        output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
        stats[counter_key] += 1

        # Выводим промежуточную статистику
        if stats[filtered_key] % 1000 == 0:
            print(f"Обработано {audio_dir}: {stats[filtered_key]} записей")


# Загружаем и обрабатываем датасеты в потоковом режиме
print("Обрабатываем тренировочный набор (train)")
train_dataset = load_dataset(
    'mozilla-foundation/common_voice_17_0',
    'ru',
    token=HF_TOKEN,
    streaming=True,
    split='train'
)
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
process_dataset(train_dataset, train_file, "train", "train", "filtered_train")

print("Обрабатываем тестовый набор (для добавления к тренировочному)")
test_dataset = load_dataset(
    'mozilla-foundation/common_voice_17_0',
    'ru',
    token=HF_TOKEN,
    streaming=True,
    split='test'
)
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
process_dataset(test_dataset, train_file, "train", "train", "filtered_train")

print("Обрабатываем валидационный набор")
val_dataset = load_dataset(
    'mozilla-foundation/common_voice_17_0',
    'ru',
    token=HF_TOKEN,
    streaming=True,
    split='validation'
)
val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))
process_dataset(val_dataset, val_file, "validate", "val", "filtered_val")

# Закрываем файлы манифестов
train_file.close()
val_file.close()

# Выводим итоговую статистику
print("\nИтоговая статистика:")
print(f"Отфильтровано для тренировки: {stats['filtered_train']} записей")
print(f"Отфильтровано для валидации: {stats['filtered_val']} записей")
print(f"\nГотово! Манифесты сохранены в:")
print(f"- ./datasets/common_voice/train_manifest.json")
print(f"- ./datasets/common_voice/validate_manifest.json")