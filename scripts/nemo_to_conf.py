# Load canary model if not previously loaded in this notebook instance
import os

from nemo.collections.asr.models import EncDecMultiTaskModel
from omegaconf import OmegaConf

if 'canary_model' not in locals():
    canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-180m-flash')

base_model_cfg = OmegaConf.load("../config/base.yaml")


base_model_cfg['name'] = 'canary-180m-flash-finetune'
base_model_cfg.pop("init_from_nemo_model", None)
base_model_cfg['init_from_pretrained_model'] = "nvidia/canary-180m-flash"
canary_model.save_tokenizers('./canary_flash_tokenizers/')
for lang in os.listdir('../canary_flash_tokenizers'):
    base_model_cfg['model']['tokenizer']['langs'][lang] = {}
    base_model_cfg['model']['tokenizer']['langs'][lang]['dir'] = os.path.join('../canary_flash_tokenizers', lang)
    base_model_cfg['model']['tokenizer']['langs'][lang]['type'] = 'bpe'
base_model_cfg['spl_tokens']['model_dir'] = os.path.join('../canary_flash_tokenizers', "spl_tokens")
base_model_cfg['model']['prompt_format'] = canary_model._cfg['prompt_format']
base_model_cfg['model']['prompt_defaults'] = canary_model._cfg['prompt_defaults']
base_model_cfg['model']['model_defaults'] = canary_model._cfg['model_defaults']
base_model_cfg['model']['preprocessor'] = canary_model._cfg['preprocessor']
base_model_cfg['model']['encoder'] = canary_model._cfg['encoder']
base_model_cfg['model']['transf_decoder'] = canary_model._cfg['transf_decoder']
base_model_cfg['model']['transf_encoder'] = canary_model._cfg['transf_encoder']
cfg = OmegaConf.create(base_model_cfg)
with open("../config/base-canary-180m-flash-finetune-ru.yaml", "w") as f:
    OmegaConf.save(cfg, f)