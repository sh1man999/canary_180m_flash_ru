from utils.normalizer_text import normalize_text

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