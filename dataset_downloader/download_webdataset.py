import json
import os

import librosa
import soundfile

import tqdm
from datasets import load_dataset, Audio, Dataset, IterableDataset

from dataset_downloader.dto import HFDatasetConversionConfig
from utils.normalizer_text import normalize_text


def prepare_output_dirs(cfg: HFDatasetConversionConfig):
    """
    Prepare output directories and subfolders as needed.
    Also prepare the arguments of the config with these directories.
    """
    output_dir = os.path.abspath(cfg.output_dir)
    output_dir = os.path.join(output_dir, cfg.path)

    if cfg.name is not None:
        output_dir = os.path.join(output_dir, cfg.name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cfg.resolved_output_dir = output_dir
    cfg.split_output_dir = None


def infer_dataset_segments(batch, cfg: HFDatasetConversionConfig):
    """
    Helper method to run in batch mode over a mapped Dataset.

    Infers the path of the subdirectories for the dataset, removing {extracted/HASH}.

    Returns:
        A cleaned list of path segments
    """
    segments = []
    segment, path = os.path.split(batch[cfg.audio_path]['path'])
    segments.insert(0, path)
    while segment not in ('', os.path.sep):
        segment, path = os.path.split(segment)
        segments.insert(0, path)

    if 'extracted' in segments:
        index_of_basedir = segments.index("extracted")
        segments = segments[(index_of_basedir + 1 + 1) :]  # skip .../extracted/{hash}/

    return segments


def replace_ext(audio_filepath):
    # replace any ext with .wav
    audio_filepath, ext = os.path.splitext(audio_filepath)
    return audio_filepath + '.wav'


def prepare_audio_filepath(audio_filepath):
    """
    Helper method to run in batch mode over a mapped Dataset.

    Prepares the audio filepath and its subdirectories. Remaps the extension to .wav file.

    Args:
        audio_filepath: String path to the audio file.

    Returns:
        Cleaned filepath renamed to be a wav file.
    """
    audio_basefilepath = os.path.split(audio_filepath)[0]
    if not os.path.exists(audio_basefilepath):
        os.makedirs(audio_basefilepath, exist_ok=True)

    # Remove temporary fmt file
    if os.path.exists(audio_filepath):
        os.remove(audio_filepath)


    # Remove previous run file
    if os.path.exists(audio_filepath):
        os.remove(audio_filepath)
    return audio_filepath


def build_map_dataset_to_nemo_func(cfg: HFDatasetConversionConfig, basedir):
    """
    Helper method to run in batch mode over a mapped Dataset.

    Creates a function that can be passed to Dataset.map() containing the config and basedir.
    Useful to map a HF dataset to NeMo compatible format in an efficient way for offline processing.

    Returns:
        A function pointer which can be used for Dataset.map()
    """

    def map_dataset_to_nemo(batch):
        # Write audio file to correct path
        segments = infer_dataset_segments(batch, cfg)
        audio_filepath = os.path.join(*segments)
        audio_filepath = replace_ext(audio_filepath)
        file_path_write = os.path.join(cfg.output_dir, cfg.split, 'audio', audio_filepath)
        batch['audio_filepath'] = os.path.join(basedir, audio_filepath)
        prepare_audio_filepath(file_path_write)
        soundfile.write(file_path_write, batch[cfg.audio_path]['array'], samplerate=cfg.sampling_rate, format='wav')
        batch['text'] = normalize_text(batch[cfg.data_path]['text'])
        batch['source_lang'] = cfg.source_lang
        batch['target_lang'] = cfg.target_lang
        batch['taskname'] = cfg.taskname
        batch['pnc'] = cfg.pnc
        batch['duration'] = librosa.get_duration(y=batch[cfg.audio_path]['array'], sr=batch[cfg.audio_path]['sampling_rate'])
        del batch[cfg.audio_path]
        del batch[cfg.data_path]
        del batch["__key__"]
        del batch["__url__"]
        return batch

    return map_dataset_to_nemo


def convert_offline_dataset_to_nemo(
        dataset: Dataset,
        cfg: HFDatasetConversionConfig,
        basedir: str,
        manifest_filepath: str,
):

    num_proc = cfg.num_proc
    if num_proc < 0:
        num_proc = max(1, os.cpu_count() // 2)

    dataset = dataset.map(build_map_dataset_to_nemo_func(cfg, basedir), num_proc=num_proc)
    ds_iter = iter(dataset)

    with open(manifest_filepath, 'w', encoding='utf-8') as manifest_f:
        for idx, sample in enumerate(
                tqdm.tqdm(
                    ds_iter, desc=f'Processing {cfg.path} (split : {cfg.split}):', total=len(dataset), unit=' samples'
                )
        ):
            manifest_f.write(f"{json.dumps(sample, ensure_ascii=cfg.ensure_ascii)}\n")


def process_dataset(dataset: IterableDataset, cfg: HFDatasetConversionConfig):
    dataset = dataset.cast_column(cfg.audio_path, Audio(cfg.sampling_rate, mono=True))
    manifest_filename = f"{cfg.split}_manifest.jsonl"
    manifest_filepath = os.path.abspath(os.path.join(cfg.output_dir, manifest_filename))
    dataset_dir = cfg.output_dir.split('/')
    dataset_dir = dataset_dir[len(dataset_dir)-1]
    dir_dataset = './' + os.path.join('datasets', dataset_dir, cfg.split, 'audio')
    convert_offline_dataset_to_nemo(dataset, cfg, basedir=dir_dataset, manifest_filepath=manifest_filepath)

    print()
    print("Dataset conversion finished !")


def download(cfg: HFDatasetConversionConfig):
    dataset = load_dataset(cfg.path,
                           cache_dir=None,
                           token=cfg.use_auth_token,
                           trust_remote_code=True
                           )
    print()
    print("Multiple splits found for dataset", cfg.path, ":", list(dataset.keys()))

    keys = list(dataset.keys())
    for key in keys:
        ds_split = dataset[key]
        print(f"Processing split {key} for dataset {cfg.path}")

        cfg.split_output_dir = os.path.join(cfg.output_dir, key)
        print(ds_split)
        cfg.split = key
        process_dataset(ds_split, cfg)

        del dataset[key], ds_split

    # reset the split output directory
    cfg.split_output_dir = None
