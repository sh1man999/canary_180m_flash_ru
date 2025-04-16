import lightning.pytorch as pl
import torch
from nemo.utils import logging
import math

from metrics.no_punct_wer import NoPunctWER


class NoPunctWERCallback(pl.Callback):
    """
    Колбэк для расчета WER без пунктуации в распределенной среде.
    Корректно обрабатывает нестандартные структуры данных NeMo.
    """

    def __init__(self):
        super().__init__()
        self.wer_no_punct = None

    def on_validation_start(self, trainer, pl_module):
        # Создаем метрику в начале валидации
        self.wer_no_punct = NoPunctWER()
        logging.info("Инициализация расчета WER без пунктуации")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Обрабатывает батч данных из валидационного набора и обновляет метрику.
        Адаптирована для работы с объектами PromptedAudioToTextMiniBatch.
        """
        try:
            # Извлекаем данные из батча в зависимости от его структуры
            if hasattr(batch, 'input_signal') and hasattr(batch, 'input_signal_length'):
                # Для PromptedAudioToTextMiniBatch
                signal = batch.input_signal
                signal_len = batch.input_signal_length
                transcript = batch.target_text_tokens
                transcript_len = batch.target_text_tokens_lengths
            elif isinstance(batch, tuple) and len(batch) >= 4:
                # Для обычных кортежей
                signal, signal_len, transcript, transcript_len = batch
            else:
                # Логируем структуру батча для диагностики
                logging.warning(f"Неизвестная структура батча: {type(batch)}")
                return

            # Получаем предсказания модели
            with torch.no_grad():
                # Используем forward модели для получения логитов
                log_probs, encoded_len = None, None

                # Проверяем сигнатуру метода forward
                if hasattr(pl_module, 'forward'):
                    try:
                        # Пробуем вызвать с именованными аргументами
                        outputs = pl_module.forward(
                            input_signal=signal,
                            input_signal_length=signal_len
                        )

                        # Распаковываем результаты в зависимости от их структуры
                        if isinstance(outputs, tuple) and len(outputs) >= 2:
                            log_probs, encoded_len = outputs[:2]
                        else:
                            # Альтернативный формат вывода
                            log_probs = outputs
                            encoded_len = signal_len
                    except Exception as e:
                        logging.warning(f"Ошибка при вызове forward: {str(e)}")
                        return

                # Если не удалось получить предсказания, используем выходы из outputs
                if log_probs is None and outputs is not None:
                    if isinstance(outputs, dict) and 'log_probs' in outputs:
                        log_probs = outputs['log_probs']
                        encoded_len = outputs.get('encoded_len', signal_len)

                # Если все еще нет данных, выходим
                if log_probs is None:
                    logging.warning("Не удалось получить предсказания из модели")
                    return

                # Получаем гипотезы и эталоны
                hypotheses = self._get_hypotheses(pl_module, log_probs, encoded_len)
                references = self._get_references(pl_module, transcript, transcript_len)

                # Обновляем метрику только если получены данные
                if hypotheses and references:
                    self.wer_no_punct.update(hypotheses, references)
                    logging.debug(f"Batch {batch_idx}: обработано {len(hypotheses)} примеров")

        except Exception as e:
            logging.warning(f"Ошибка при обработке батча {batch_idx}: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())

    def _get_hypotheses(self, pl_module, log_probs, encoded_len):
        """
        Извлекает гипотезы из логитов модели.
        Адаптируется к различным API декодирования в NeMo.
        """
        try:
            # Проверяем различные варианты доступа к декодеру
            decoder = None
            if hasattr(pl_module, 'wer') and hasattr(pl_module.wer, 'decoding'):
                decoder = pl_module.wer.decoding
            elif hasattr(pl_module, 'decoding'):
                decoder = pl_module.decoding
            elif hasattr(pl_module, '_wer') and hasattr(pl_module._wer, 'decoding'):
                decoder = pl_module._wer.decoding

            if decoder is None:
                logging.warning("Не найден декодер для преобразования логитов")
                return []

            # Пробуем различные методы декодирования в зависимости от версии NeMo
            if hasattr(decoder, 'ctc_decoder_predictions_tensor'):
                best_hyps = decoder.ctc_decoder_predictions_tensor(
                    log_probs, encoded_len, return_hypotheses=True
                )
                # Извлекаем текст из гипотез
                return [hyp.text if hasattr(hyp, 'text') else str(hyp) for hyp in best_hyps]
            elif hasattr(decoder, 'decode_with_confidence'):
                best_hyps = decoder.decode_with_confidence(log_probs, encoded_len)
                return [hyp[0] for hyp in best_hyps]  # Берем текст, исключая confidence score
            else:
                # Пробуем прямое декодирование через argmax и токенизатор
                predictions = log_probs.argmax(dim=-1, keepdim=False).cpu().numpy()
                texts = []
                for i, pred in enumerate(predictions):
                    length = encoded_len[i].item() if encoded_len.dim() > 0 else encoded_len.item()
                    text = self._decode_tokens(pl_module, pred[:length])
                    texts.append(text)
                return texts

        except Exception as e:
            logging.warning(f"Ошибка при получении гипотез: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
            return []

    def _get_references(self, pl_module, transcript, transcript_len):
        """
        Извлекает эталонные тексты из транскрипций.
        """
        try:
            # Выбираем правильный декодер
            decoder = None
            if hasattr(pl_module, 'wer') and hasattr(pl_module.wer, 'decoding'):
                decoder = pl_module.wer.decoding
            elif hasattr(pl_module, 'decoding'):
                decoder = pl_module.decoding
            elif hasattr(pl_module, '_wer') and hasattr(pl_module._wer, 'decoding'):
                decoder = pl_module._wer.decoding

            if decoder is None:
                logging.warning("Не найден декодер для преобразования эталонов")
                return []

            # Преобразуем эталонные транскрипции из токенов в текст
            references = []
            for idx in range(transcript.size(0)):
                target_len = transcript_len[idx].item()
                target_tokens = transcript[idx][:target_len].cpu().numpy().tolist()

                # Пробуем разные методы декодирования
                if hasattr(decoder, 'decode_tokens_to_str'):
                    ref_text = decoder.decode_tokens_to_str(target_tokens)
                elif hasattr(decoder, 'ids_to_text'):
                    ref_text = decoder.ids_to_text(target_tokens)
                else:
                    ref_text = self._decode_tokens(pl_module, target_tokens)

                references.append(ref_text)

            return references

        except Exception as e:
            logging.warning(f"Ошибка при получении эталонов: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
            return []

    def _decode_tokens(self, pl_module, tokens):
        """
        Универсальный метод декодирования токенов в текст.
        Пробует различные API, доступные в модели.
        """
        # Пробуем различные возможные API для декодирования
        if hasattr(pl_module, 'tokenizer'):
            return pl_module.tokenizer.ids_to_text(tokens)
        elif hasattr(pl_module, 'decoder'):
            return pl_module.decoder.decode(tokens)
        elif hasattr(pl_module, 'processor'):
            return pl_module.processor.batch_decode([tokens])[0]
        else:
            # Простой фоллбэк, просто соединяем токены
            return ' '.join(map(str, tokens))

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Вычисляет и логирует метрику WER без пунктуации
        в конце эпохи валидации.
        """
        try:
            # Проверяем, есть ли данные для расчета
            if self.wer_no_punct is None or self.wer_no_punct.total.item() == 0:
                logging.warning("Недостаточно данных для расчёта WER без пунктуации")
                return

            # Вычисляем значение метрики
            wer_no_punct_value = self.wer_no_punct.compute()

            # Логируем метрику
            pl_module.log('val_wer_no_punct', wer_no_punct_value, prog_bar=True, sync_dist=True)

            # Добавляем в callback_metrics для EarlyStopping
            trainer.callback_metrics["val_wer_no_punct"] = wer_no_punct_value

            # Выводим в лог
            self._print_metrics(pl_module, wer_no_punct_value)

        except Exception as e:
            logging.error(f"Ошибка при расчёте WER без пунктуации: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

    def _print_metrics(self, pl_module, wer_no_punct):
        """Форматирует и выводит метрики в лог"""

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
        original_wer = None
        if hasattr(pl_module, 'wer') and hasattr(pl_module.wer, 'compute'):
            try:
                original_wer = pl_module.wer.compute()
                if isinstance(original_wer, tuple):
                    original_wer = original_wer[0]
            except:
                pass

        # Красивый вывод в лог
        logging.info("╔════════════════════════════════════════════════════════════╗")
        logging.info("║             МЕТРИКИ КАЧЕСТВА РАСПОЗНАВАНИЯ                 ║")
        logging.info("╠════════════════════════════════════════════════════════════╣")

        if original_wer is not None:
            logging.info(f"║ WER с пунктуацией:  {format_wer(original_wer):<35}  ║")

        logging.info(f"║ WER без пунктуации: {format_wer(wer_no_punct):<35}  ║")
        logging.info("╠════════════════════════════════════════════════════════════╣")

        if hasattr(self.wer_no_punct, 'examples'):
            logging.info(f"║ Обработано примеров:{self.wer_no_punct.examples.item():<35}  ║")
        elif hasattr(self.wer_no_punct, 'total'):
            logging.info(f"║ Всего слов в эталоне:{self.wer_no_punct.total.item():<33}  ║")

        logging.info("╚════════════════════════════════════════════════════════════╝")