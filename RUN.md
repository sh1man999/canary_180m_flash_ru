## Запуск обучения и тестирования модели

После успешной подготовки всех компонентов можно переходить к обучению и тестированию модели.

### Запуск обучения

```bash
python speech_to_text_aed.py \
    --config-path=./ \
    --config-name=russian_finetune \
    model.train_ds.manifest_filepath=manifests/train_manifest.json \
    model.validation_ds.manifest_filepath=manifests/val_manifest.json \
    model.test_ds.manifest_filepath=manifests/test_manifest.json \
    spl_tokens.model_dir=./tokenizers_spl \
    model.tokenizer.langs.spl_tokens.dir=./tokenizers_spl \
    model.tokenizer.langs.ru.dir=./tokenizers_ru/tokenizer_spe_bpe_v1024 \
    trainer.devices=1 \
    exp_manager.exp_dir=./russian_experiments \
    exp_manager.resume_if_exists=False \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.create_wandb_logger=False \
    exp_manager.exp_dir="canary_results" \
    exp_manager.resume_ignore_no_checkpoint=true \
    trainer.max_steps=75000 \
    trainer.log_every_n_steps=100 \
    model.transf_decoder.config_dict.num_layers=6 \
    model.transf_decoder.config_dict.ffn_dropout=0.2 \
    model.optim.sched.warmup_steps=5000 \
    trainer.accumulate_grad_batches=4
```

> **Примечание**: Обратите внимание на добавленные параметры для улучшения обучения с нуля: увеличенное количество слоёв декодера (6), повышенный dropout (0.2), увеличенное количество шагов разогрева (5000) и накопление градиентов (4).

### Запуск отдельного тестирования

Если вам нужно протестировать уже обученную модель:

```bash
python speech_to_text_aed.py \
    --config-path=./ \
    --config-name=russian_finetune \
    fit=False \
    model.train_ds.manifest_filepath=manifests/train_manifest.json \
    model.validation_ds.manifest_filepath=manifests/val_manifest.json \
    model.test_ds.manifest_filepath=manifests/test_manifest.json \
    spl_tokens.model_dir=./tokenizers_spl \
    model.tokenizer.langs.spl_tokens.dir=./tokenizers_spl \
    model.tokenizer.langs.ru.dir=./tokenizers_ru/tokenizer_spe_bpe_v1024 \
    trainer.devices=1 \
    exp_manager.exp_dir=./russian_experiments \
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.create_wandb_logger=False \
    exp_manager.exp_dir="canary_results" \
    exp_manager.resume_ignore_no_checkpoint=true \
    +init_from_pretrained_model=./russian_experiments/canary-small-ru/2025-03-22_05-11-50/checkpoints/canary-small-ru.nemo
```

> **Важно**: При запуске тестирования установите параметр `fit=False` и укажите путь к сохранённой модели через `+init_from_pretrained_model`.

Мониторинг обучения можно осуществлять через TensorBoard, запустив команду:
```bash
tensorboard --logdir=./canary_results
```