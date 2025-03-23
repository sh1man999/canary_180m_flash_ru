## Запуск обучения и тестирования модели

После успешной подготовки всех компонентов можно переходить к обучению и тестированию модели.

### Запуск обучения

```bash
python scripts/speech_to_text_aed.py \
    --config-path=../config \
    --config-name=canary-180m-flash-finetune.yaml \
    name="canary-180m-flash-finetune" \
    exp_manager.create_wandb_logger=False \
    exp_manager.exp_dir="canary_results" \
    exp_manager.resume_ignore_no_checkpoint=true \
    trainer.max_steps=10 \
    trainer.log_every_n_steps=1
```

### Запуск отдельного тестирования

Если вам нужно протестировать уже обученную модель:

```bash
python speech_to_text_aed.py \
    --config-path=./config \
    --config-name=canary-180m-flash-finetune \
    fit=False \
    model.test_ds.manifest_filepath=manifests/test_manifest.json \
    exp_manager.create_wandb_logger=False \
    +init_from_nemo_model=./canary_results/canary-small-ru/2025-03-22_17-43-54/checkpoints/canary-small-ru.nemo
```

> **Важно**: При запуске тестирования установите параметр `fit=False` и укажите путь к сохранённой модели через `+init_from_pretrained_model`.

Мониторинг обучения можно осуществлять через TensorBoard, запустив команду:
```bash
tensorboard --logdir=./canary_results
```