import yaml
from nemo.collections.asr.models import EncDecMultiTaskModel
from omegaconf import OmegaConf


def main():
    canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-180m-flash')
    # Получаем конфигурацию модели
    model_config = canary_model.cfg

    # Преобразуем конфигурацию в словарь и затем в YAML
    config_dict = OmegaConf.to_container(model_config, resolve=True)

    # Сохраняем в файл
    with open('canary_180m_flash_config.yaml', 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    #output = canary_model.transcribe(['test.wav'])
    #print(output[0].text)

if __name__ == "__main__":
    main()
