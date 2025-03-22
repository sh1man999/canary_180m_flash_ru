# Подготовка русскоязычной модели Canary ASR

В этом документе описаны шаги по подготовке и обучению модели распознавания речи для русского языка на базе архитектуры Canary.

## Шаг 1: Подготовка датасета

Выполните скрипт для скачивания и обработки речевых данных:

```bash
python postprocess_dataset.py
```

Этот скрипт подготовит необходимые манифесты и аудиофайлы для последующего обучения.

## Шаг 2: Создание русского токенизатора

Если у вас еще нет токенизатора для русского языка, необходимо его создать:

```bash
# Предварительная обработка текстового корпуса
python pretokenize.py

# Создание BPE-токенизатора
python process_asr_text_tokenizer.py \
  --data_file=corpus/russian_corpus.txt \
  --vocab_size=1024 \
  --data_root=./tokenizers_ru \
  --tokenizer="spe" \
  --spe_type=bpe \
  --spe_character_coverage=1.0 \
  --no_lower_case \
  --log
```

## Шаг 3: Создание специального токенизатора

Создайте токенизатор для обработки служебных токенов:

```bash
python build_canary_2_special_tokenizer.py tokenizers_spl
```

Данный токенизатор обрабатывает метки языков и команды для модели, такие как `translate`, `transcribe`, `ru` и другие.

---

После выполнения этих шагов можно переходить к настройке и запуску процесса обучения модели.