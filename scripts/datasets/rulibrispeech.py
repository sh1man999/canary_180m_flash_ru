import os

from entrypoint.config import BASE_DIR
from dataset_downloader.download_webdataset import download
from dataset_downloader.dto import HFDatasetConversionConfig

if __name__ == '__main__':
    cfg = HFDatasetConversionConfig(
        path="Sh1man/rulibrispeech",
        output_dir=os.path.join(BASE_DIR, 'datasets', 'rulibrispeech'),
    )
    download(cfg)