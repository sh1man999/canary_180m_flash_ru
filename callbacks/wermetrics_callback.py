import lightning.pytorch as pl
import torch
from nemo.utils import logging

from metrics.no_punct_wer import NoPunctWER


class ASRModelWithCustomMetrics(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Инициализируем кастомную метрику WER без пунктуации
        self.wer_no_punct = NoPunctWER()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Выполняем стандартную логику валидации
        outputs = super().validation_step(batch, batch_idx, dataloader_idx)

        # Извлекаем предсказания и эталоны для кастомной метрики
        signal, signal_len, transcript, transcript_len = batch

        # Получаем предсказания модели
        with torch.no_grad():
            log_probs, encoded_len, *_ = self.forward(
                input_signal=signal,
                input_signal_length=signal_len
            )

            # Преобразуем предсказания в токены с помощью декодера
            predictions = self.wer.decoding.ctc_decoder_predictions_tensor(
                log_probs, encoded_len, return_hypotheses=True
            )

            # Получаем текстовые гипотезы
            hypotheses = [hyp.text if hasattr(hyp, 'text') else hyp for hyp in predictions]

            # Преобразуем эталоны из токенов в текст
            references = []
            for idx in range(transcript.size(0)):
                target_len = transcript_len[idx].item()
                target_tokens = transcript[idx][:target_len].cpu().numpy().tolist()
                ref_text = self.wer.decoding.decode_tokens_to_str(target_tokens)
                references.append(ref_text)

            # Обновляем метрику без пунктуации
            self.wer_no_punct.update(hypotheses, references)

        return outputs

    def on_validation_epoch_end(self):
        # Вызываем оригинальную логику
        super().on_validation_epoch_end()

        # Вычисляем значение метрики без пунктуации
        wer_no_punct_value = self.wer_no_punct.compute()

        # Логируем метрику (синхронизация уже обеспечена метрикой)
        self.log('val_wer_no_punct', wer_no_punct_value, prog_bar=True)

        # Добавляем в callback_metrics для EarlyStopping
        self.trainer.callback_metrics["val_wer_no_punct"] = wer_no_punct_value

        # Форматируем вывод для лога
        self._print_metrics(wer_no_punct_value)

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