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
        self.hypotheses_buffer = []
        self.references_buffer = []
        self.world_size = 1

    def _clean_text(self, text):
        """Унифицированная функция очистки текста от пунктуации"""
        if not isinstance(text, str):
            return text
        # Удаляем пунктуацию и приводим к нижнему регистру
        cleaned = self.punctuation_pattern.sub('', text).lower()
        # Удаляем лишние пробелы
        return re.sub(r'\s+', ' ', cleaned).strip()

    def on_validation_start(self, trainer, pl_module):
        """Инициализация в начале валидации"""
        self.hypotheses_buffer = []
        self.references_buffer = []
        # Получаем размер распределённого мира
        if hasattr(trainer, 'world_size'):
            self.world_size = trainer.world_size

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Перехватываем результаты сразу после вычисления на каждом батче"""
        # Напрямую получаем данные из метрики WER
        if hasattr(pl_module, 'wer') and hasattr(pl_module.wer, 'decoding'):
            # Получаем гипотезы и ссылки из текущего батча
            hyps, refs = self._extract_batch_data(pl_module, batch)

            if hyps and refs:
                # Добавляем в локальные буферы для этого ранка
                self.hypotheses_buffer.extend(hyps)
                self.references_buffer.extend(refs)

    def _extract_batch_data(self, pl_module, batch):
        """Извлекаем данные напрямую из батча и модели"""
        try:
            # Получаем входные данные из батча
            signal, signal_len, transcript, transcript_len = batch

            # Делаем вывод модели без градиентов
            with torch.no_grad():
                # Получаем предсказания модели
                log_probs, encoded_len, *_ = pl_module.forward(
                    input_signal=signal,
                    input_signal_length=signal_len
                )

                # Преобразуем предсказания в токены с помощью декодера модели
                best_hyps = pl_module.wer.decoding.ctc_decoder_predictions_tensor(
                    log_probs, encoded_len, return_hypotheses=True
                )

                # Получаем текстовые гипотезы
                hypotheses = [hyp.text if hasattr(hyp, 'text') else hyp for hyp in best_hyps]

                # Преобразуем эталонные транскрипции из токенов в текст
                references = []
                for idx in range(transcript.size(0)):
                    target_len = transcript_len[idx].item()
                    target_tokens = transcript[idx][:target_len].cpu().numpy().tolist()
                    ref_text = pl_module.wer.decoding.decode_tokens_to_str(target_tokens)
                    references.append(ref_text)

                return hypotheses, references
        except Exception as e:
            logging.debug(f"Ошибка извлечения данных из батча: {str(e)}")
            return [], []

    def on_validation_epoch_end(self, trainer, pl_module):
        """Собираем данные со всех ранков и вычисляем метрику"""
        try:
            # Синхронизируем списки между всеми ранками
            all_hypotheses = self._gather_distributed_data(trainer, self.hypotheses_buffer)
            all_references = self._gather_distributed_data(trainer, self.references_buffer)

            # Вычисляем метрики только на мастер-ранке
            if trainer.is_global_zero and all_hypotheses and all_references:
                # Обычный WER с пунктуацией
                original_wer = word_error_rate(all_hypotheses, all_references)

                # WER без пунктуации
                cleaned_hypotheses = [self._clean_text(hyp) for hyp in all_hypotheses]
                cleaned_references = [self._clean_text(ref) for ref in all_references]
                wer_no_punct = word_error_rate(cleaned_hypotheses, cleaned_references)

                # Логируем метрики
                pl_module.log('val_wer', original_wer, on_epoch=True, sync_dist=False)
                pl_module.log('val_wer_no_punct', wer_no_punct, on_epoch=True, sync_dist=False)

                # Добавляем в метрики для EarlyStopping
                trainer.callback_metrics["val_wer_no_punct"] = torch.tensor(wer_no_punct)

                # Красивый вывод метрик (как в вашем оригинальном коде)
                self._print_metrics(original_wer, wer_no_punct, len(all_references))

        except Exception as e:
            logging.error(f"Ошибка при расчёте WER без пунктуации: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

    def _gather_distributed_data(self, trainer, local_data):
        """Собирает данные со всех ранков в распределённом режиме"""
        if self.world_size <= 1:
            return local_data

        # Преобразуем в тензор для синхронизации
        # Обратите внимание: это упрощённый подход
        # В реальном коде нужно использовать более сложную логику
        # для сериализации/десериализации строк
        try:
            gathered_data = trainer.strategy.all_gather(local_data)
            # Если собранные данные имеют размерности, спрямляем их
            if isinstance(gathered_data, torch.Tensor) and gathered_data.dim() > 1:
                return gathered_data.flatten().tolist()
            elif isinstance(gathered_data, list):
                # Если результат - список списков, соединяем их
                flat_data = []
                for item in gathered_data:
                    if isinstance(item, list):
                        flat_data.extend(item)
                    else:
                        flat_data.append(item)
                return flat_data
            else:
                return gathered_data
        except Exception as e:
            logging.warning(f"Ошибка при сборе распределённых данных: {str(e)}")
            return local_data

    def _print_metrics(self, original_wer, wer_no_punct, num_examples):
        """Форматирует и выводит метрики в лог"""

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
        logging.info(f"║ Обработано примеров:{num_examples:<35}  ║")
        logging.info("╚════════════════════════════════════════════════════════════╝")