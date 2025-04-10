import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from config import BASE_DIR


def combine_manifests(exclude_datasets=None):
    """
    Объединяет манифесты из всех датасетов в папке datasets,
    исключая указанные датасеты.

    Args:
        exclude_datasets (list): Список имен датасетов для исключения
    """
    if exclude_datasets is None:
        exclude_datasets = []

    # Путь к директории datasets
    datasets_dir = os.path.join(BASE_DIR, 'datasets')

    # Словарь для группировки манифестов по типам (train, validate, test)
    manifest_groups = defaultdict(list)

    # Сканируем все поддиректории в datasets
    for dataset_name in os.listdir(datasets_dir):
        dataset_path = os.path.join(datasets_dir, dataset_name)

        # Пропускаем, если это не директория или датасет в списке исключений
        if not os.path.isdir(dataset_path) or dataset_name in exclude_datasets:
            continue

        # Ищем все манифесты в директории датасета
        for file_name in os.listdir(dataset_path):
            if file_name.endswith('_manifest.jsonl'):
                # Определяем тип манифеста (train, validate, test)
                manifest_type = file_name.split('_')[0]
                manifest_path = os.path.join(dataset_path, file_name)

                manifest_groups[manifest_type].append((dataset_name, manifest_path))

    # Объединяем и сохраняем манифесты по типам
    for manifest_type, manifest_files in manifest_groups.items():
        output_path = os.path.join(datasets_dir, f'{manifest_type}_manifest.jsonl')

        print(f"Объединение {len(manifest_files)} {manifest_type} манифестов...")

        # Счетчик общего количества записей
        total_entries = 0

        with open(output_path, 'w', encoding='utf-8') as out_file:
            for dataset_name, manifest_path in manifest_files:
                entries_count = 0

                # Читаем и записываем каждую строку из исходного манифеста
                with open(manifest_path, 'r', encoding='utf-8') as in_file:
                    for line in tqdm(in_file, desc=f"Обработка {dataset_name}"):
                        out_file.write(line)
                        entries_count += 1
                        total_entries += 1

                print(f"  - {dataset_name}: {entries_count} записей")

        print(f"Объединенный {manifest_type}_manifest.jsonl содержит {total_entries} записей")
        print(f"Сохранено в {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Объединение манифестов из разных датасетов')
    parser.add_argument('--exclude', nargs='+', help='Датасеты для исключения из объединения')

    args = parser.parse_args()

    combine_manifests(exclude_datasets=args.exclude)

    print("Объединение манифестов завершено!")