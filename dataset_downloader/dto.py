from dataclasses import dataclass
from typing import Optional


@dataclass
class HFDatasetConversionConfig:
    # Nemo Dataset info
    output_dir: str  # path to output directory where the files will be saved

    # HF Dataset info
    path: str  # HF dataset path
    name: Optional[str] = None  # name of the dataset subset
    split: Optional[str] = None  # split of the dataset subset
    use_auth_token: bool = False  # whether authentication token should be passed or not (Required for MCV)

    # NeMo dataset conversion
    sampling_rate: int = 16000
    streaming: bool = False  # Whether to use Streaming dataset API. [NOT RECOMMENDED]
    num_proc: int = -1
    ensure_ascii: bool = False  # When saving the JSON entry, whether to ensure ascii.
    source_lang: str = 'ru'
    target_lang: str = 'ru'
    taskname: str = 'asr'
    pnc = 'yes' # пунктуация
    audio_path: str = 'mp3'
    data_path: str = 'json'

    split_output_dir: Optional[str] = None
