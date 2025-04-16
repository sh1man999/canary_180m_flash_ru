import re
from typing import List

import torch
from torchmetrics import Metric


class NoPunctWER(Metric):
    """
    Высокоэффективная метрика Word Error Rate без учёта пунктуации.

    Реализует расчёт WER с автоматической очисткой от знаков пунктуации,
    полностью совместимый с распределёнными вычислениями через механизмы TorchMetrics.
    """

    def __init__(
            self,
            dist_sync_on_step: bool = False,
            punctuation_pattern: str = r'[.,!?:;«»()\-]'
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Определяем состояния метрики с корректными функциями редукции
        self.add_state("error_sum", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("word_count", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("sample_count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

        # Компилируем регулярное выражение для оптимизации производительности
        self.punctuation_pattern = re.compile(punctuation_pattern)

    def update(self, predictions: List[str], references: List[str]) -> None:
        """
        Обновляет метрику новыми данными.

        Args:
            predictions: Список предсказанных текстов
            references: Список эталонных текстов
        """
        # Валидация входных данных
        if not predictions or not references:
            return

        # Выравнивание размерностей
        if len(predictions) != len(references):
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]

        # Очистка текстов от пунктуации
        cleaned_predictions = [self._normalize_text(text) for text in predictions]
        cleaned_references = [self._normalize_text(text) for text in references]

        # Расчёт статистики по каждой паре текстов
        batch_errors = 0.0
        batch_words = 0.0

        for pred, ref in zip(cleaned_predictions, cleaned_references):
            # Разбиение на слова
            pred_words = pred.split()
            ref_words = ref.split()

            # Расчёт расстояния редактирования (оптимизированный алгоритм)
            edit_distance = self._levenshtein_distance(pred_words, ref_words)

            # Накопление статистики
            batch_errors += edit_distance
            batch_words += len(ref_words)

        # Обновление состояний метрики тензорами для корректного распределённого вычисления
        self.error_sum += torch.tensor(batch_errors, dtype=torch.float64)
        self.word_count += torch.tensor(batch_words, dtype=torch.float64)
        self.sample_count += torch.tensor(len(predictions), dtype=torch.int64)

    def compute(self) -> torch.Tensor:
        """
        Вычисляет финальное значение метрики.

        Returns:
            torch.Tensor: Значение WER без пунктуации
        """
        # Защита от деления на ноль
        if self.word_count == 0:
            return torch.tensor(float('inf'), dtype=torch.float64)

        # Расчёт WER как отношения суммы ошибок к общему количеству слов
        return self.error_sum / self.word_count

    def _normalize_text(self, text: str) -> str:
        """
        Нормализует текст: удаляет пунктуацию, приводит к нижнему регистру,
        стандартизирует пробельные символы.

        Args:
            text: Исходный текст

        Returns:
            str: Нормализованный текст
        """
        if not isinstance(text, str):
            return ""

        # Удаление пунктуации
        cleaned = self.punctuation_pattern.sub('', text)

        # Приведение к нижнему регистру
        cleaned = cleaned.lower()

        # Нормализация пробелов (замена всех последовательностей пробельных символов на один пробел)
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Удаление начальных и конечных пробелов
        return cleaned.strip()

    def _levenshtein_distance(self, source: List[str], target: List[str]) -> int:
        """
        Вычисляет расстояние Левенштейна между двумя последовательностями слов.
        Использует оптимизированный алгоритм с минимальным потреблением памяти.

        Args:
            source: Исходная последовательность слов
            target: Целевая последовательность слов

        Returns:
            int: Расстояние Левенштейна
        """
        # Оптимизация граничных случаев
        if len(source) == 0:
            return len(target)
        if len(target) == 0:
            return len(source)

        # Оптимизация по памяти: храним только две строки матрицы
        previous_row = list(range(len(target) + 1))
        current_row = [0] * (len(target) + 1)

        for i in range(1, len(source) + 1):
            # Инициализация первого элемента текущей строки
            current_row[0] = i

            for j in range(1, len(target) + 1):
                # Расчёт стоимости операций редактирования
                deletion_cost = previous_row[j] + 1
                insertion_cost = current_row[j - 1] + 1
                substitution_cost = previous_row[j - 1] + (0 if source[i - 1] == target[j - 1] else 1)

                # Выбор операции с минимальной стоимостью
                current_row[j] = min(deletion_cost, insertion_cost, substitution_cost)

            # Обмен строк для следующей итерации
            previous_row, current_row = current_row, previous_row

        # После последней итерации результат находится в previous_row
        return previous_row[len(target)]
