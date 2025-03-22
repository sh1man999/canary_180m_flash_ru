1. # Подготовка датасета (Скачивает датасет и подготавливает к обучению)
`python postprocess_dataset.py`

2. # Создаем ru токенизатор (Если нету)
`python pretokenize.py`
`python process_asr_text_tokenizer.py \
  --data_file=corpus/russian_corpus.txt \
  --vocab_size=1024 \
  --data_root=./tokenizers_ru \
  --tokenizer="spe" \
  --spe_type=bpe \
  --spe_character_coverage=1.0 \
  --no_lower_case \
  --log`

3. # Создаём токенизатор spl(Если нету) (пока хз для чего он)
`python build_canary_2_special_tokenizer.py tokenizers_spl`

