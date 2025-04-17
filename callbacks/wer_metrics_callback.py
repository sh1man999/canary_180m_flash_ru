from metrics.no_punct_wer import NoPunctWER
import lightning.pytorch as pl
import math
from nemo.utils import logging


class NoPunctWERCallback(pl.Callback):
    """
    Коллбэк Lightning для вычисления Word Error Rate без пунктуации.
    """
    def __init__(self, punctuation_pattern: str = r'[.,!?:;«»()\-]'):
        super().__init__()
        self.wer_metric = NoPunctWER(dist_sync_on_step=False, punctuation_pattern=punctuation_pattern)
        self.val_predictions = []
        self.val_references = []
        self.test_predictions = []
        self.test_references = []

    def format_wer(self, wer_value):
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

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0):
        if hasattr(outputs, 'preds') and hasattr(batch, 'tokens'):
            preds = pl_module.decode(outputs['preds'])
            refs = [pl_module.tokenizer.ids_to_text(token_ids) for token_ids in batch.tokens]
            self.val_predictions.extend(preds)
            self.val_references.extend(refs)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.val_predictions and self.val_references:
            self.wer_metric.update(self.val_predictions, self.val_references)
            wer_no_punct_value = self.wer_metric.compute()
            pl_module.log('val_wer_no_punct', wer_no_punct_value, sync_dist=True)

            # Попытка получить оригинальный WER, если он логируется моделью
            original_wer_value = pl_module.trainer.callback_metrics.get('val_wer', float('inf'))

            cleaned_references = [self.wer_metric._normalize_text(ref) for ref in self.val_references]

            # Красивый вывод в лог
            logging.info("╔════════════════════════════════════════════════════════════╗")
            logging.info("║             МЕТРИКИ КАЧЕСТВА РАСПОЗНАВАНИЯ                 ║")
            logging.info("╠════════════════════════════════════════════════════════════╣")
            logging.info(f"║ WER с пунктуацией:  {self.format_wer(original_wer_value):<35}  ║")
            logging.info(f"║ WER без пунктуации: {self.format_wer(wer_no_punct_value):<35}  ║")
            logging.info("╠════════════════════════════════════════════════════════════╣")
            logging.info(f"║ Обработано примеров:{len(cleaned_references):<35}  ║")
            logging.info("╚════════════════════════════════════════════════════════════╝")

            self.wer_metric.reset()
            self.val_predictions = []
            self.val_references = []

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0):
        if hasattr(outputs, 'preds') and hasattr(batch, 'tokens'):
            preds = pl_module.decode(outputs['preds'])
            refs = [pl_module.tokenizer.ids_to_text(token_ids) for token_ids in batch.tokens]
            self.test_predictions.extend(preds)
            self.test_references.extend(refs)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.test_predictions and self.test_references:
            self.wer_metric.update(self.test_predictions, self.test_references)
            wer_no_punct_value = self.wer_metric.compute()
            pl_module.log('test_wer_no_punct', wer_no_punct_value, sync_dist=True)

            # Аналогично для тестовой выборки, если нужно
            original_wer_value = pl_module.trainer.callback_metrics.get('test_wer', float('inf'))
            cleaned_references = [self.wer_metric._normalize_text(ref) for ref in self.test_references]

            logging.info("╔════════════════════════════════════════════════════════════╗")
            logging.info("║             МЕТРИКИ КАЧЕСТВА РАСПОЗНАВАНИЯ (ТЕСТ)           ║")
            logging.info("╠════════════════════════════════════════════════════════════╣")
            logging.info(f"║ WER с пунктуацией:  {self.format_wer(original_wer_value):<35}  ║")
            logging.info(f"║ WER без пунктуации: {self.format_wer(wer_no_punct_value):<35}  ║")
            logging.info("╠════════════════════════════════════════════════════════════╣")
            logging.info(f"║ Обработано примеров:{len(cleaned_references):<35}  ║")
            logging.info("╚════════════════════════════════════════════════════════════╝")

            self.wer_metric.reset()
            self.test_predictions = []
            self.test_references = []