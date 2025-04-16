import re
import torch
from typing import List, Optional, Union
from torchmetrics import Metric


class NoPunctWER(Metric):
    """
    Высокоэффективная метрика Word Error Rate без учета пунктуации.

    Ключевые особенности:
    1. Автоматическая очистка от знаков пунктуации перед расчетом
    2. Полная поддержка распределенных вычислений через встроенные механизмы TorchMetrics
    3. Оптимизированный алгоритм расчета расстояния Левенштейна
    4. Детальная статистика для мониторинга процесса распознавания
    """

    def __init__(
            self,
            dist_sync_on_step: bool = False,
            punctuation_pattern: str = r'[.,!?:;«»()\-]'
    ):
        """
        Инициализирует метрику WER без пунктуации.

        Args:
            dist_sync_on_step: Синхронизировать значения на каждом шаге (для распределенных вычислений)
            punctuation_pattern: Регулярное выражение для удаления знаков пунктуации
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Регистрируем состояния, которые будут синхронизироваться в распределенном режиме
        self.add_state("errors", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total_words", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total_examples", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

        # Компилируем регулярное выражение для повышения производительности
        self.punctuation_pattern = re.compile(punctuation_pattern)

    def update(self, predictions: List[str], references: List[str]) -> None:
        """
        Обновляет метрику новыми данными.

        Args:
            predictions: Список предсказанных текстов
            references: Список эталонных текстов
        """
        # Проверка валидности входных данных
        if not predictions or not references:
            return

        if len(predictions) != len(references):
            # Обрезаем до одинаковой длины в случае несоответствия
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]

        # Очищаем тексты от пунктуации и нормализуем
        cleaned_predictions = [self._preprocess_text(text) for text in predictions]
        cleaned_references = [self._preprocess_text(text) for text in references]

        # Рассчитываем ошибки и статистику для каждой пары текстов
        batch_errors = 0.0
        batch_words = 0.0

        for pred, ref in zip(cleaned_predictions, cleaned_references):
            # Разбиваем на слова
            pred_words = pred.split()
            ref_words = ref.split()

            # Рассчитываем расстояние Левенштейна на уровне слов
            distance = self._compute_levenshtein(pred_words, ref_words)

            # Накапливаем статистику
            batch_errors += distance
            batch_words += len(ref_words)

        # Обновляем состояния метрики
        self.errors += torch.tensor(batch_errors, dtype=torch.float64)
        self.total_words += torch.tensor(batch_words, dtype=torch.float64)
        self.total_examples += torch.tensor(len(predictions), dtype=torch.int64)

    def compute(self) -> torch.Tensor:
        """
        Вычисляет финальное значение метрики.

        Returns:
            torch.Tensor: Значение WER без пунктуации
        """
        if self.total_words == 0:
            return torch.tensor(float('inf'))

        return self.errors / self.total_words

    def _preprocess_text(self, text: str) -> str:
        """
        Предобрабатывает текст: удаляет пунктуацию, приводит к нижнему регистру,
        нормализует пробелы.

        Args:
            text: Исходный текст

        Returns:
            str: Очищенный и нормализованный текст
        """
        if not isinstance(text, str):
            return ""

        # Удаляем пунктуацию
        cleaned = self.punctuation_pattern.sub('', text)

        # Приводим к нижнему регистру
        cleaned = cleaned.lower()

        # Нормализуем пробелы (заменяем множественные пробелы на один)
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Убираем начальные и конечные пробелы
        return cleaned.strip()

    def _compute_levenshtein(self, source: List[str], target: List[str]) -> int:
        """
        Вычисляет расстояние Левенштейна между двумя последовательностями.
        Оптимизированная реализация с минимальным использованием памяти.

        Args:
            source: Исходная последовательность слов
            target: Целевая последовательность слов

        Returns:
            int: Расстояние Левенштейна
        """
        # Оптимизация для граничных случаев
        if len(source) == 0:
            return len(target)
        if len(target) == 0:
            return len(source)

        # Оптимизация по памяти: храним только две смежные строки матрицы
        previous_row = list(range(len(target) + 1))
        current_row = [0] * (len(target) + 1)

        for i in range(1, len(source) + 1):
            # Инициализируем первый элемент текущей строки
            current_row[0] = i

            for j in range(1, len(target) + 1):
                # Вычисляем стоимость операций
                deletion = previous_row[j] + 1
                insertion = current_row[j - 1] + 1
                substitution = previous_row[j - 1] + (0 if source[i - 1] == target[j - 1] else 1)

                # Выбираем операцию с минимальной стоимостью
                current_row[j] = min(deletion, insertion, substitution)

            # Меняем строки местами для следующей итерации
            previous_row, current_row = current_row, previous_row

        # После последней итерации результат находится в previous_row
        return previous_row[len(target)]    