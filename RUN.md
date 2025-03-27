## Подготовка датасета

### В доке написано что для эффективного использования GPU при обучении, нужно использовать oomptimizer 
### Пункт Pushing GPU utilization to the limits with bucketing and OOMptimizer
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/datasets.html
```bash
python scripts/estimate_duration_bins_2d.py \
    --prompt-format canary2 \
    --prompt "[{'role':'user','slots':{'source_lang':'<|ru|>','target_lang':'<|ru|>','pnc':'<|pnc|>','itn':'<|noitn|>','diarize':'<|nodiarize|>','timestamp':'<|notimestamp|>','emotion':'<|emo:undefined|>','decodercontext':''}}]" \
    --tokenizer canary_flash_tokenizers/spl_tokens/tokenizer.model canary_flash_tokenizers/en/tokenizer.model canary_flash_tokenizers/de/tokenizer.model canary_flash_tokenizers/es/tokenizer.model canary_flash_tokenizers/fr/tokenizer.model canary_flash_tokenizers/ru/tokenizer.model \
    --langs spl_tokens en de es fr ru \
    --buckets 30 \
    --sub-buckets 5 \
    ./datasets/train_manifest.json
```
После выполнения скрипта выше, получаем bucket_duration_bins. Далее прокидываем его в --buckets

```bash
python scripts/oomptimizer.py \
    --config-path ./config/canary-180m-flash-finetune-ru_oomptimizer.yaml \
    --module-name nemo.collections.asr.models.EncDecMultiTaskModel \
    --memory-fraction 0.90 \
    --buckets '[[2.808,17],[2.808,19],[2.808,21],[2.808,23],[2.808,34],[3.192,21],[3.192,23],[3.192,25],[3.192,27],[3.192,39],[3.456,22],[3.456,25],[3.456,27],[3.456,30],[3.456,39],[3.744,24],[3.744,27],[3.744,29],[3.744,32],[3.744,46],[3.960,26],[3.960,29],[3.960,31],[3.960,35],[3.960,50],[4.176,27],[4.176,30],[4.176,33],[4.176,36],[4.176,51],[4.392,29],[4.392,33],[4.392,36],[4.392,39],[4.392,53],[4.584,30],[4.584,33],[4.584,37],[4.584,40],[4.584,53],[4.788,32],[4.788,35],[4.788,39],[4.788,43],[4.788,59],[4.968,33],[4.968,37],[4.968,40],[4.968,45],[4.968,59],[5.148,34],[5.148,38],[5.148,41],[5.148,46],[5.148,60],[5.304,36],[5.304,39],[5.304,43],[5.304,46],[5.304,60],[5.496,37],[5.496,41],[5.496,45],[5.496,49],[5.496,65],[5.664,38],[5.664,42],[5.664,46],[5.664,50],[5.664,63],[5.808,40],[5.808,43],[5.808,47],[5.808,51],[5.808,70],[5.976,41],[5.976,45],[5.976,49],[5.976,53],[5.976,67],[6.156,42],[6.156,47],[6.156,50],[6.156,55],[6.156,68],[6.336,42],[6.336,47],[6.336,51],[6.336,55],[6.336,69],[6.504,45],[6.504,50],[6.504,53],[6.504,57],[6.504,70],[6.696,44],[6.696,49],[6.696,53],[6.696,58],[6.696,73],[6.864,46],[6.864,51],[6.864,54],[6.864,59],[6.864,72],[7.056,48],[7.056,52],[7.056,56],[7.056,61],[7.056,83],[7.236,49],[7.236,54],[7.236,58],[7.236,62],[7.236,76],[7.464,50],[7.464,55],[7.464,59],[7.464,63],[7.464,78],[7.728,52],[7.728,57],[7.728,60],[7.728,64],[7.728,85],[7.992,53],[7.992,57],[7.992,61],[7.992,66],[7.992,84],[8.304,54],[8.304,59],[8.304,63],[8.304,68],[8.304,93],[8.676,54],[8.676,59],[8.676,64],[8.676,69],[8.676,88],[9.240,56],[9.240,62],[9.240,66],[9.240,71],[9.240,89],[11.484,59],[11.484,65],[11.484,69],[11.484,74],[11.484,90]]'
```


## Запуск обучения и тестирования модели

После успешной подготовки всех компонентов можно переходить к обучению и тестированию модели.

### Запуск обучения с добавлением нового языка (Если новый язык уже добавлен в модель, то второй вариант использовать !)

```bash
python train.py \
    --config-path=./config \
    --config-name=canary-180m-flash-finetune-ru.yaml \
    name="canary-180m-flash-finetune" \
    exp_manager.create_wandb_logger=False \
    exp_manager.exp_dir="canary_results" \
    exp_manager.resume_ignore_no_checkpoint=true \
    trainer.max_steps=60000 \
    trainer.log_every_n_steps=100 \
    init_from_pretrained_model.model0.name="nvidia/canary-180m-flash" \
    init_from_pretrained_model.model0.exclude=[\"transf_decoder._embedding.token_embedding\",\"log_softmax.mlp.layer0\"]
```

### Запуск обучения с контрольной точки без oomptimizer

```bash
python train.py \
    --config-path=./config \
    --config-name=canary-180m-flash-finetune-ru.yaml \
    name="canary-180m-flash-finetune" \
    exp_manager.create_wandb_logger=False \
    exp_manager.exp_dir="canary_results" \
    exp_manager.resume_ignore_no_checkpoint=true \
    trainer.max_steps=60000 \
    trainer.log_every_n_steps=100 \
    init_from_ptl_ckpt="./models/last.ckpt"
    
```
### Запуск обучения с контрольной точки c oomptimizer

```bash
python train.py \
    --config-path=./config \
    --config-name=canary-180m-flash-finetune-ru_oomptimizer.yaml \
    name="canary-180m-flash-finetune" \
    exp_manager.create_wandb_logger=False \
    exp_manager.exp_dir="canary_results" \
    exp_manager.resume_ignore_no_checkpoint=true \
    trainer.max_steps=60000 \
    trainer.log_every_n_steps=100 \
    init_from_ptl_ckpt="./models/last.ckpt"
```

### Запуск отдельного тестирования

Если вам нужно протестировать уже обученную модель:

```bash
python train.py \
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