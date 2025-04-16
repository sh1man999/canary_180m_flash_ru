import torch
import re
from torchmetrics import Metric


class NoPunctWER(Metric):
    """
    Оптимизированная метрика WER без пунктуации, корректно работающая
    в распределенной среде. Автоматически синхронизирует данные между ранками.
    """

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # Определяем состояния метрики с правильными функциями редукции
        self.add_state("errors", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("examples", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

        # Компилируем регулярное выражение для оптимальной производительности
        self.punctuation_pattern = re.compile(r'[.,!?:;«»()\-]')

    def update(self, preds, targets):
        """
        Обновляет метрику новыми данными. Автоматически обрабатывает
        очистку текста от пунктуации и вычисление расстояния Левенштейна.

        Args:
            preds (List[str]): Список предсказанных текстов
            targets (List[str]): Список эталонных текстов
        """
        # Очищаем тексты от пунктуации
        cleaned_preds = [self._clean_text(p) for p in preds]
        cleaned_targets = [self._clean_text(t) for t in targets]

        # Вычисляем метрику
        batch_errors, batch_words = 0, 0
        for pred, target in zip(cleaned_preds, cleaned_targets):
            # Разбиваем на слова
            pred_words = pred.split()
            target_words = target.split()

            # Вычисляем расстояние Левенштейна
            edit_distance = self._levenshtein_distance(pred_words, target_words)

            # Аккумулируем результаты
            batch_errors += edit_distance
            batch_words += len(target_words)

        # Обновляем состояние метрики
        self.errors += torch.tensor(batch_errors, dtype=torch.float32)
        self.total += torch.tensor(batch_words, dtype=torch.float32)
        self.examples += torch.tensor(len(preds), dtype=torch.int64)

    def compute(self):
        """
        Вычисляет итоговое значение метрики.

        Returns:
            torch.Tensor: Значение WER без пунктуации
        """
        return self.errors / self.total if self.total > 0 else torch.tensor(float('inf'))

    def _clean_text(self, text):
        """
        Очищает текст от пунктуации и нормализует пробелы.

        Args:
            text (str): Исходный текст

        Returns:
            str: Очищенный текст
        """
        if not isinstance(text, str):
            return ""
        # Удаляем пунктуацию и приводим к нижнему регистру
        cleaned = self.punctuation_pattern.sub('', text).lower()
        # Нормализуем пробелы
        return re.sub(r'\s+', ' ', cleaned).strip()

    def _levenshtein_distance(self, source, target):
        """
        Вычисляет расстояние Левенштейна между двумя последовательностями.
        Оптимизированная реализация с использованием матрицы динамического
        программирования.

        Args:
            source (List): Исходная последовательность слов
            target (List): Целевая последовательность слов

        Returns:
            int: Расстояние Левенштейна
        """
        # Создаем матрицу размером (len(source)+1) x (len(target)+1)
        distance_matrix = [[0 for _ in range(len(target) + 1)] for _ in range(len(source) + 1)]

        # Инициализируем первую строку и первый столбец
        for i in range(1, len(source) + 1):
            distance_matrix[i][0] = i

        for j in range(1, len(target) + 1):
            distance_matrix[0][j] = j

        # Заполняем матрицу
        for i in range(1, len(source) + 1):
            for j in range(1, len(target) + 1):
                # Стоимость замены
                substitution_cost = 0 if source[i - 1] == target[j - 1] else 1

                # Выбираем минимальную операцию
                distance_matrix[i][j] = min(
                    distance_matrix[i - 1][j] + 1,  # Удаление
                    distance_matrix[i][j - 1] + 1,  # Вставка
                    distance_matrix[i - 1][j - 1] + substitution_cost  # Замена или совпадение
                )

        # Возвращаем итоговое расстояние Левенштейна
        return distance_matrix[len(source)][len(target)]