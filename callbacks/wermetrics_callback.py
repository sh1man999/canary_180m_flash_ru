import math
import re
import torch
import lightning.pytorch as pl
from nemo.utils import logging
from nemo.collections.asr.metrics.wer import word_error_rate


class WERWithoutPunctuation(pl.Callback):
    def __init__(self):
        super().__init__()
        self.punctuation_pattern = re.compile(r'[.,!?:;«»()\-]')
        self.predictions = []
        self.references = []

    def _clean_text(self, text):
        """Унифицированная функция очистки текста от пунктуации"""
        if not isinstance(text, str):
            return text
        # Удаляем пунктуацию и приводим к нижнему регистру
        cleaned = self.punctuation_pattern.sub('', text).lower()
        # Удаляем лишние пробелы
        return re.sub(r'\s+', ' ', cleaned).strip()

    def on_validation_start(self, trainer, pl_module):
        # Очищаем коллекции в начале валидации
        self.predictions = []
        self.references = []

        # Важно: НЕ заменяем метод, а просто подписываемся на результаты
        if hasattr(pl_module, 'wer') and hasattr(pl_module.wer, 'compute'):
            original_compute = pl_module.wer.compute

            def compute_hook(*args, **kwargs):
                # Получаем оригинальный результат
                result = original_compute(*args, **kwargs)

                # Собираем предсказания и эталоны через официальные методы API
                # Это важно: мы не изменяем логику, а только собираем результаты
                if hasattr(pl_module.wer, '_predictions') and hasattr(pl_module.wer, '_references'):
                    # Используем синхронизированный вызов для сбора данных со всех ранков
                    all_preds = trainer.strategy.all_gather(pl_module.wer._predictions)
                    all_refs = trainer.strategy.all_gather(pl_module.wer._references)

                    # Обрабатываем только на мастер-ранке для избежания дублирования
                    if trainer.is_global_zero:
                        self.predictions.extend(all_preds)
                        self.references.extend(all_refs)

                return result

            # Сохраняем оригинальный метод
            self._original_compute = original_compute
            pl_module.wer.compute = compute_hook

    def on_validation_epoch_end(self, trainer, pl_module):
        """Рассчитываем WER без пунктуации и выводим обе метрики в понятном формате"""
        try:
            # Важно: восстанавливаем оригинальный метод
            if self._original_compute is not None and hasattr(pl_module, 'wer'):
                pl_module.wer.compute = self._original_compute

            # Выполняем расчёт только на мастер-ранке
            if not trainer.is_global_zero:
                return

            if not self.predictions or not self.references:
                logging.warning("Недостаточно данных для расчёта WER без пунктуации")
                return

            # Рассчитываем оригинальный WER
            original_wer = word_error_rate(self.predictions, self.references)

            # Очищаем тексты от пунктуации
            cleaned_predictions = [self._clean_text(pred) for pred in self.predictions]
            cleaned_references = [self._clean_text(ref) for ref in self.references]

            # Рассчитываем WER без пунктуации
            wer_no_punct = word_error_rate(cleaned_predictions, cleaned_references)

            # Логируем метрику без пунктуации — НА ВСЕХ РАНКАХ ОДИНАКОВО
            # sync_dist=False, т.к. мы уже агрегировали данные
            pl_module.log('val_wer_no_punct', wer_no_punct, on_epoch=True, prog_bar=True, sync_dist=False)

            # Добавляем в словарь метрик для EarlyStopping
            trainer.callback_metrics["val_wer_no_punct"] = torch.tensor(wer_no_punct)

            # Подготовка понятного вывода метрик
            def format_wer(wer_value):
                """Форматирует WER для человекочитаемого вывода"""
                if math.isinf(wer_value):
                    return "∞ (несопоставимые данные)"
                elif wer_value > 1.0:
                    return f"{wer_value:.4f} ({wer_value * 100:.1f}%) - критически высокий"
                elif wer_value > 0.5:
                    return f"{wer_value:.4f} ({wer_value * 100:.1f}%) - очень высокий"
                elif wer_value > 0.3:
                    return f"{wer_value:.4f} ({wer_value * 100:.1f}%) - высокий"
                elif wer_value > 0.15:
                    return f"{wer_value:.4f} ({wer_value * 100:.1f}%) - средний"
                else:
                    return f"{wer_value:.4f} ({wer_value * 100:.1f}%) - низкий"

            # Красивый вывод в лог
            logging.info("╔════════════════════════════════════════════════════════════╗")
            logging.info("║             МЕТРИКИ КАЧЕСТВА РАСПОЗНАВАНИЯ                 ║")
            logging.info("╠════════════════════════════════════════════════════════════╣")
            logging.info(f"║ WER с пунктуацией:  {format_wer(original_wer):<35}  ║")
            logging.info(f"║ WER без пунктуации: {format_wer(wer_no_punct):<35}  ║")
            logging.info("╠════════════════════════════════════════════════════════════╣")
            logging.info(f"║ Обработано примеров:{len(cleaned_references):<35}  ║")
            logging.info("╚════════════════════════════════════════════════════════════╝")

        except Exception as e:
            logging.error(f"Ошибка при расчёте WER без пунктуации: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())