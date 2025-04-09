import re


def normalize_text(text: str) -> str:
    # Базовая очистка
    text = (text.replace("(", "")
            .replace(")", "")
            .replace(" .", ".")
            .replace("«", "")
            .replace("»", "")
            #.replace("\"", "")
            .replace("'", "")
            .replace(" ,", ",")
            .replace(" :", ":")
            .replace(" ;", ";")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace("  ", " ")
            .replace("„", "")
            .replace("„", "“")
            .strip()
            )

    # Нормализация многоточий с учётом контекста
    # Заменяем юникод-символ многоточия
    text = text.replace("…", "...")

    # Исправляем двухточечное многоточие, но только если это не часть трёхточечного
    text = re.sub(r'(?<![.])\.\.(?!\.)', '...', text)

    # Исправляем две точки после вопросительного/восклицательного знаков
    text = re.sub(r'([?!])\.\.(?!\.)', r'\1...', text)

    # Удаляем двойные/тройные кавычки
    text = re.sub(r'"{2,}', '"', text)

    # Обрабатываем несбалансированные кавычки
    # Если количество кавычек нечетное, удаляем последнюю
    if text.count('"') % 2 != 0:
        last_quote_index = text.rfind('"')
        if last_quote_index != -1:
            text = text[:last_quote_index] + text[last_quote_index + 1:]

    return text


if __name__ == '__main__':
    test_cases = [
        '"Я все время слышу слова ""застой"", ""тупик"", ""затор"", ""блокировка""."',
        '"""Отсадите от меня этого остолопа!"" - не унимался ученик с последней парты."',
    ]

    for i, tc in enumerate(test_cases, 1):
        result = normalize_text(tc)
        print(f"Тест {i}:")
        print(f"Исходный: {tc}")
        print(f"Результат: {result}")
        print("---")