import glob
import json
import os
from entrypoint.config import BASE_DIR


def main():
    corpus_dir = os.path.join(BASE_DIR, "corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    manifest_path = os.path.join(os.path.join(BASE_DIR, 'datasets'))
    manifest_files = glob.glob(os.path.join(manifest_path, '*.jsonl'))
    with open(os.path.join(corpus_dir, 'russian_corpus.txt'), "w", encoding="utf-8") as f:
        for manifest_file in manifest_files:
            with open(manifest_file) as reader:
                line = reader.readline()
                data = json.loads(line)
                f.write(data['text']+'\n')


if __name__ == "__main__":
    main()