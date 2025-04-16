import torch
import re
from torchmetrics import Metric


class NoPunctWER(Metric):
    """
    Высокопроизводительная метрика WER без пунктуации, оптимизированная для
    распределенного выполнения в PyTorch Lightning.
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
        Обновляет метрику новыми данными.

        Args:
            preds (List[str]): Список предсказанных текстов
            targets (List[str]): Список эталонных текстов
        """
        # Проверка валидности входных данных
        if not preds or not targets or len(preds) != len(targets):
            return

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
            distance = self._levenshtein_distance(pred_words, target_words)

            # Аккумулируем результаты
            batch_errors += distance
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
        Оптимизированная реализация с использованием динамического программирования.

        Args:
            source (List): Исходная последовательность слов
            target (List): Целевая последовательность слов

        Returns:
            int: Расстояние Левенштейна
        """
        # Оптимизация для пустых последовательностей
        if len(source) == 0:
            return len(target)
        if len(target) == 0:
            return len(source)

        # Оптимизация памяти: храним только текущую и предыдущую строки
        # вместо всей матрицы расстояний
        previous_row = range(len(target) + 1)
        current_row = [0] * (len(target) + 1)

        for i in range(1, len(source) + 1):
            # Инициализация первого элемента текущей строки
            current_row[0] = i

            for j in range(1, len(target) + 1):
                # Вычисляем стоимость операций
                deletion = previous_row[j] + 1
                insertion = current_row[j - 1] + 1
                substitution = previous_row[j - 1]

                # Если слова не совпадают, увеличиваем стоимость замены
                if source[i - 1] != target[j - 1]:
                    substitution += 1

                # Выбираем минимальную стоимость операции
                current_row[j] = min(deletion, insertion, substitution)

            # Обновляем предыдущую строку для следующей итерации
            previous_row, current_row = current_row, previous_row

        # В последнем обмене previous_row содержит актуальные значения
        return previous_row[len(target)]