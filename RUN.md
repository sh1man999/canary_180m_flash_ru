## Подготовка к обучению


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
    ./datasets/train_manifest.jsonl
```
После выполнения скрипта выше, получаем bucket_duration_bins. Далее прокидываем его в --buckets

```bash
python scripts/oomptimizer.py \
    --config-path ./configs/canary-180m-flash-finetune-ru_oomptimizer.yaml \
    --module-name nemo.collections.asr.models.EncDecMultiTaskModel \
    --memory-fraction 0.90 \
    --buckets '[[2.664,18],[2.664,21],[2.664,24],[2.664,27],[2.664,42],[3.060,21],[3.060,24],[3.060,27],[3.060,30],[3.060,44],[3.384,24],[3.384,27],[3.384,29],[3.384,32],[3.384,47],[3.660,26],[3.660,28],[3.660,31],[3.660,34],[3.660,48],[3.930,27],[3.930,30],[3.930,33],[3.930,36],[3.930,50],[4.176,29],[4.176,32],[4.176,35],[4.176,38],[4.176,52],[4.416,31],[4.416,34],[4.416,37],[4.416,40],[4.416,56],[4.632,32],[4.632,36],[4.632,38],[4.632,42],[4.632,54],[4.850,34],[4.850,38],[4.850,40],[4.850,44],[4.850,59],[5.060,35],[5.060,39],[5.060,41],[5.060,45],[5.060,60],[5.256,37],[5.256,40],[5.256,43],[5.256,46],[5.256,62],[5.472,38],[5.472,42],[5.472,45],[5.472,48],[5.472,67],[5.664,40],[5.664,43],[5.664,46],[5.664,49],[5.664,64],[5.868,41],[5.868,44],[5.868,48],[5.868,51],[5.868,67],[6.080,43],[6.080,46],[6.080,49],[6.080,53],[6.080,68],[6.280,44],[6.280,48],[6.280,51],[6.280,54],[6.280,74],[6.490,45],[6.490,48],[6.490,52],[6.490,56],[6.490,74],[6.696,46],[6.696,50],[6.696,53],[6.696,57],[6.696,73],[6.936,48],[6.936,52],[6.936,55],[6.936,59],[6.936,76],[7.176,49],[7.176,53],[7.176,56],[7.176,60],[7.176,80],[7.420,50],[7.420,54],[7.420,58],[7.420,62],[7.420,79],[7.728,52],[7.728,56],[7.728,60],[7.728,64],[7.728,82],[8.040,53],[8.040,57],[8.040,60],[8.040,65],[8.040,82],[8.440,55],[8.440,59],[8.440,63],[8.440,67],[8.440,87],[8.920,57],[8.920,61],[8.920,65],[8.920,70],[8.920,90],[9.560,59],[9.560,64],[9.560,68],[9.560,73],[9.560,96],[10.536,63],[10.536,69],[10.536,74],[10.536,80],[10.536,102],[12.240,71],[12.240,78],[12.240,84],[12.240,90],[12.240,120],[14.760,83],[14.760,91],[14.760,97],[14.760,104],[14.760,138],[20.000,103],[20.000,112],[20.000,120],[20.000,130],[20.000,174]]'
```


## Запуск обучения и тестирования модели

После успешной подготовки всех компонентов можно переходить к обучению и тестированию модели.

### Запуск обучения с добавлением нового языка (Если новый язык уже добавлен в модель, то второй вариант использовать !)

```bash
python train.py \
    --config-path=./configs \
    --config-name=canary-180m-flash-finetune-ru_oomptimizer.yaml \
    name="canary-180m-flash-finetune" \
    exp_manager.create_wandb_logger=False \
    exp_manager.exp_dir="canary_results" \
    exp_manager.resume_ignore_no_checkpoint=true \
    trainer.max_steps=100000 \
    trainer.log_every_n_steps=1000 \
    init_from_pretrained_model.model0.name="nvidia/canary-180m-flash" \
    init_from_pretrained_model.model0.exclude=[\"transf_decoder._embedding.token_embedding\",\"log_softmax.mlp.layer0\"]
```

### Запуск обучения с контрольной точки без oomptimizer

```bash
python train.py \
    --config-path=./configs \
    --config-name=canary-180m-flash-finetune-ru.yaml \
    name="canary-180m-flash-finetune" \
    exp_manager.create_wandb_logger=False \
    exp_manager.exp_dir="canary_results" \
    exp_manager.resume_ignore_no_checkpoint=true \
    trainer.max_steps=60000 \
    trainer.log_every_n_steps=1000 \
    init_from_ptl_ckpt="./models/last.ckpt"
    
```
### Запуск обучения с контрольной точки c oomptimizer

```bash
python train.py \
    --config-path=./configs \
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
    --config-path=./configs \
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