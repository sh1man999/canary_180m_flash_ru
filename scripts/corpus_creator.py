import glob
import json
import os
from entrypoint.config import BASE_DIR


def main():
    """Извлекает уникальные текстовые данные из JSONL-манифестов и формирует корпус."""
    corpus_dir = os.path.join(BASE_DIR, "corpus")
    os.makedirs(corpus_dir, exist_ok=True)

    manifest_path = os.path.join(BASE_DIR, 'datasets')
    manifest_files = glob.glob(os.path.join(manifest_path, '*.jsonl'))

    output_path = os.path.join(corpus_dir, 'russian_corpus.txt')

    # Счетчики для аналитики процесса
    stats = {
        'processed_files': 0,
        'processed_entries': 0,
        'unique_texts': 0,
        'duplicates': 0,
        'errors': 0
    }

    # Эффективное хранение уникальных текстов с помощью множества
    unique_texts = set()

    with open(output_path, "w", encoding="utf-8") as output_file:
        for manifest_file in manifest_files:
            stats['processed_files'] += 1

            with open(manifest_file, 'r', encoding='utf-8') as reader:
                for line_number, line in enumerate(reader, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        if 'text' in data and data['text']:
                            text = data['text'].strip()
                            stats['processed_entries'] += 1

                            # Проверка на дубликаты
                            if text not in unique_texts:
                                unique_texts.add(text)
                                output_file.write(text + '\n')
                                stats['unique_texts'] += 1
                            else:
                                stats['duplicates'] += 1
                        else:
                            stats['errors'] += 1
                    except json.JSONDecodeError:
                        stats['errors'] += 1

    # Детальный отчет о выполнении для аналитики
    print(f"Обработка корпуса завершена:")
    print(f"  ✓ Обработано файлов: {stats['processed_files']}")
    print(f"  ✓ Проанализировано записей: {stats['processed_entries']}")
    print(f"  ✓ Уникальных текстов сохранено: {stats['unique_texts']}")
    print(f"  ✓ Найдено дубликатов: {stats['duplicates']}")
    print(f"  ✓ Ошибок обработки: {stats['errors']}")
    print(f"  ✓ Корпус сохранен в: {output_path}")

    # Краткое описание эффективности дедупликации
    if stats['duplicates'] > 0:
        dedup_rate = (stats['duplicates'] / stats['processed_entries']) * 100
        print(f"  ✓ Уровень дедупликации: {dedup_rate:.2f}% ({stats['duplicates']} повторов удалено)")


if __name__ == "__main__":
    main()