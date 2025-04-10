# install ubuntu 24 dependencies
sudo apt-get install portaudio19-dev

# Подготовка русскоязычной модели Canary ASR

В этом документе описаны шаги по подготовке и обучению модели распознавания речи для русского языка на базе архитектуры Canary.

## Шаг 1: Подготовка датасета

Выполните скрипт для скачивания датасета и обработки речевых данных:
```bash
python scripts/datasets/common_voice_21.py
python scripts/datasets/rulibrispeech.py
```

Объединение датасетов
Этот скрипт подготовит необходимые манифесты и аудиофайлы для последующего обучения.
```bash
python scripts/datasets/compare_datasets.py
```

## Шаг 2: Создание русского токенизатора

Если у вас еще нет токенизатора для русского языка, необходимо его создать:
Мы используем стандартный алгоритм побайтового кодирования (Byte-pair encoding) со словарями размером 128, 512 и 1024 токенов.
Мы обнаружили, что словарь из 128 токенов лучше всего работает для относительно небольшого набора данных на Esperanto (примерно 250 часов).
Для более крупных наборов данных можно получить лучшие результаты с большим размером словаря (512–1024 BPE-токенов).

```bash
# Предварительная обработка текстового корпуса
python scripts/corpus_creator.py

# Создание BPE-токенизатора
python scripts/process_asr_text_tokenizer.py \
  --data_file=corpus/russian_corpus.txt \
  --vocab_size=512 \
  --data_root=./tokenizers_ru \
  --tokenizer="spe" \
  --spe_type=bpe \
  --spe_character_coverage=1.0 \
  --no_lower_case \
  --log
```
# Закинуть токенизатор в canary_flash_tokenizers

---

После выполнения этих шагов можно переходить к настройке и запуску процесса обучения модели. Описано в RUN.md