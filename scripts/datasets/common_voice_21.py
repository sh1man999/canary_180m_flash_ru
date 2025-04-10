import os

from entrypoint.config import BASE_DIR
from dataset_downloader.download_webdataset import download
from dataset_downloader.dto import HFDatasetConversionConfig

if __name__ == '__main__':
    cfg = HFDatasetConversionConfig(
        path="Sh1man/common_voice_21_rus",
        output_dir=os.path.join(BASE_DIR, 'datasets', 'common_voice_21_rus'),
    )
    download(cfg)