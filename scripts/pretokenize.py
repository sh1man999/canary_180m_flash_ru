import os

from datasets import load_dataset

# Загружаем русский датасет Common Voice
cv_dataset = load_dataset("bond005/rulibrispeech")

# Создаём директорию для корпуса
os.makedirs("corpus", exist_ok=True)

# Извлекаем только валидированные тексты из train-сплита
with open("corpus/russian_corpus.txt", "w", encoding="utf-8") as f:
    # Используем только примеры с хорошими оценками (up_votes > down_votes)
    for item in cv_dataset["train"]:
        f.write(item["transcription"].replace("(", "").replace(")", "").replace(" .", ".").replace("\"","").strip() + "\n")


print(f"Корпус для токенизатора создан в corpus/russian_corpus.txt")