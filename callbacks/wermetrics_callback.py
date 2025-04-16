import lightning.pytorch as pl
import torch
from nemo.utils import logging

from metrics.no_punct_wer import NoPunctWER


class NoPunctWERCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.wer_no_punct = None

    def on_validation_start(self, trainer, pl_module):
        # Создаем метрику в начале валидации
        self.wer_no_punct = NoPunctWER()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Та же логика сбора данных из batch, что и выше...
        signal, signal_len, transcript, transcript_len = batch

        # Получаем предсказания и эталоны
        with torch.no_grad():
            log_probs, encoded_len, *_ = pl_module.forward(
                input_signal=signal,
                input_signal_length=signal_len
            )

            # Получаем гипотезы и эталоны
            hypotheses = self._get_hypotheses(pl_module, log_probs, encoded_len)
            references = self._get_references(pl_module, transcript, transcript_len)

            # Обновляем метрику
            self.wer_no_punct.update(hypotheses, references)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Вычисляем и логируем метрику
        wer_no_punct_value = self.wer_no_punct.compute()
        pl_module.log('val_wer_no_punct', wer_no_punct_value, prog_bar=True)
        trainer.callback_metrics["val_wer_no_punct"] = wer_no_punct_value

        # Выводим в лог
        self._print_metrics(pl_module, wer_no_punct_value)

    def _print_metrics(self, wer_no_punct):
        """Форматирует и выводит метрики в лог"""
        import math

        def format_wer(wer_value):
            if isinstance(wer_value, torch.Tensor):
                wer_value = wer_value.item()

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

        # Получаем стандартный WER, если доступен
        if hasattr(self, 'wer'):
            try:
                original_wer = self.wer.compute().item()

                # Красивый вывод в лог
                logging.info("╔════════════════════════════════════════════════════════════╗")
                logging.info("║             МЕТРИКИ КАЧЕСТВА РАСПОЗНАВАНИЯ                 ║")
                logging.info("╠════════════════════════════════════════════════════════════╣")
                logging.info(f"║ WER с пунктуацией:  {format_wer(original_wer):<35}  ║")
                logging.info(f"║ WER без пунктуации: {format_wer(wer_no_punct):<35}  ║")
                logging.info("╠════════════════════════════════════════════════════════════╣")
                logging.info(f"║ Обработано примеров:{self.wer_no_punct.examples.item():<35}  ║")
                logging.info("╚════════════════════════════════════════════════════════════╝")
            except Exception as e:
                logging.warning(f"Не удалось вывести подробную статистику: {str(e)}")
                logging.info(f"WER без пунктуации: {format_wer(wer_no_punct)}")