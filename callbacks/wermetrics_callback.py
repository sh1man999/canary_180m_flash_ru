import lightning.pytorch as pl
from nemo.utils import logging

from metrics.no_punct_wer import NoPunctWER


class NoPunctWERCallback(pl.Callback):
    """
    Колбэк для интеграции NoPunctWER с моделями NeMo ASR.
    Адаптирован для работы с нестандартными форматами данных, включая PromptedAudioToTextMiniBatch.
    """

    def __init__(self):
        super().__init__()
        self.metric = None

    def on_validation_start(self, trainer, pl_module):
        """Инициализация метрики в начале валидации."""
        self.metric = NoPunctWER(dist_sync_on_step=False)
        # Перемещаем метрику на то же устройство, что и модель
        if hasattr(pl_module, 'device'):
            self.metric = self.metric.to(pl_module.device)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Извлекает предсказания и эталоны из различных форматов данных."""
        try:
            # Извлекаем данные
            predictions, references = self._extract_data(batch, outputs, pl_module)

            if predictions and references:
                # Обновляем метрику
                self.metric.update(predictions, references)

        except Exception as e:
            logging.warning(f"Ошибка обработки батча {batch_idx}: {str(e)}")

    def _extract_data(self, batch, outputs, pl_module):
        """
        Универсальный экстрактор данных, поддерживающий различные форматы NeMo.
        """
        # Проверяем возможные источники предсказаний
        predictions = self._extract_predictions(outputs, pl_module)

        # Проверяем возможные источники эталонов
        references = self._extract_references(batch, outputs, pl_module)

        return predictions, references

    def _extract_predictions(self, outputs, pl_module):
        """Извлекает предсказания из выходов модели."""
        # Имплементация из предыдущего колбэка...

    def _extract_references(self, batch, outputs, pl_module):
        """Извлекает эталоны из данных."""
        # Имплементация из предыдущего колбэка...

    def on_validation_epoch_end(self, trainer, pl_module):
        """Вычисляет и логирует финальное значение метрики."""
        if self.metric is None or self.metric.total_words == 0:
            logging.warning("Недостаточно данных для расчёта WER без пунктуации")
            return

        # Вычисляем метрику
        wer_no_punct = self.metric.compute()

        # Логируем результат
        pl_module.log('val_wer_no_punct', wer_no_punct, prog_bar=True, sync_dist=True)

        # Добавляем в метрики для EarlyStopping
        trainer.callback_metrics["val_wer_no_punct"] = wer_no_punct

        # Выводим детальную статистику
        self._print_metrics(wer_no_punct, pl_module)

    def _print_metrics(self, wer_no_punct, pl_module):
        """Форматирует и выводит результаты в лог."""
        # Имплементация из предыдущего колбэка...