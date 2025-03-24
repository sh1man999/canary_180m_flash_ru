import math
import re
import torch
import lightning.pytorch as pl
from nemo.utils import logging
from nemo.collections.asr.metrics.wer import word_error_rate
#from torchmetrics.text import WordErrorRate


class WERWithoutPunctuation(pl.Callback):
    def __init__(self):
        super().__init__()
        self.punctuation_pattern = re.compile(r'[.,!?:;«»()\-]')
        self.predictions = []
        self.references = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Сохраняем данные о предсказаниях и эталонах
        if hasattr(pl_module, 'wer') and hasattr(pl_module.wer, 'decoding'):
            # Получаем последние предсказания и эталоны
            if hasattr(pl_module.wer, 'last_batch_hypotheses'):
                self.predictions.extend(pl_module.wer.last_batch_hypotheses)
            if hasattr(pl_module.wer, 'last_batch_references'):
                self.references.extend(pl_module.wer.last_batch_references)

    def on_validation_epoch_start(self, trainer, pl_module):
        # Очищаем списки в начале эпохи
        self.predictions = []
        self.references = []

        # Монки-патчим метод update для сохранения данных
        if hasattr(pl_module, 'wer'):
            original_update = pl_module.wer.update

            def patched_update(predictions, predictions_lengths, targets, targets_lengths,
                               predictions_mask=None, input_ids=None):
                # Вызываем оригинальный метод
                result = original_update(predictions, predictions_lengths, targets,
                                         targets_lengths, predictions_mask, input_ids)

                # Сохраняем последние обработанные данные
                decoding = pl_module.wer.decoding
                with torch.no_grad():
                    tgt_lenths_cpu_tensor = targets_lengths.long().cpu()
                    targets_cpu_tensor = targets.long().cpu()

                    # Проверяем batch_dim_index
                    if pl_module.wer.batch_dim_index != 0:
                        targets_cpu_tensor = targets_cpu_tensor.transpose(0, pl_module.wer.batch_dim_index)

                    # Собираем ссылки
                    references = []
                    for ind in range(targets_cpu_tensor.shape[0]):
                        tgt_len = tgt_lenths_cpu_tensor[ind].item()
                        target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
                        reference = decoding.decode_tokens_to_str(target)
                        references.append(reference)

                    # Получаем гипотезы
                    hypotheses = pl_module.wer.decode(predictions, predictions_lengths,
                                                      predictions_mask, input_ids, targets)

                # Сохраняем данные для расчёта WER без пунктуации
                pl_module.wer.last_batch_hypotheses = [h.text if hasattr(h, 'text') else h for h in hypotheses]
                pl_module.wer.last_batch_references = references

                return result

            # Заменяем метод
            pl_module.wer.update = patched_update

    def on_validation_epoch_end(self, trainer, pl_module):
        """Рассчитываем WER без пунктуации и выводим обе метрики в понятном формате"""
        try:
            if not self.predictions or not self.references:
                logging.warning("Недостаточно данных для расчёта WER без пунктуации")
                return

            # Рассчитываем оригинальный WER
            original_wer = word_error_rate(self.predictions, self.references)

            # Очищаем тексты от пунктуации
            cleaned_predictions = []
            cleaned_references = []

            for pred in self.predictions:
                if isinstance(pred, str):
                    # Удаляем пунктуацию и приводим к нижнему регистру
                    cleaned_pred = self.punctuation_pattern.sub('', pred).lower()
                    # Удаляем лишние пробелы
                    cleaned_pred = re.sub(r'\s+', ' ', cleaned_pred).strip()
                    cleaned_predictions.append(cleaned_pred)

            for ref in self.references:
                if isinstance(ref, str):
                    # Удаляем пунктуацию и приводим к нижнему регистру
                    cleaned_ref = self.punctuation_pattern.sub('', ref).lower()
                    # Удаляем лишние пробелы
                    cleaned_ref = re.sub(r'\s+', ' ', cleaned_ref).strip()
                    cleaned_references.append(cleaned_ref)

            # Рассчитываем WER без пунктуации
            wer_no_punct = word_error_rate(cleaned_predictions, cleaned_references)

            # Логируем метрику без пунктуации
            pl_module.log('val_wer_no_punct', wer_no_punct, on_epoch=True, prog_bar=True, sync_dist=True)

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
            logging.info(f"║ WER с пунктуацией:    {format_wer(original_wer):<35} ║")
            logging.info(f"║ WER без пунктуации:   {format_wer(wer_no_punct):<35} ║")
            logging.info("╠════════════════════════════════════════════════════════════╣")
            logging.info(f"║ Обработано примеров:  {len(cleaned_references):<35}  ║")
            logging.info("╚════════════════════════════════════════════════════════════╝")

        except Exception as e:
            logging.error(f"Ошибка при расчёте WER без пунктуации: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())