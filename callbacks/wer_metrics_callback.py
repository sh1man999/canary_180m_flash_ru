import lightning.pytorch as pl
from nemo.utils import logging
import torch
from typing import List, Optional, Tuple


class NoPunctWERAdapter(pl.Callback):
    """
    Адаптер для интеграции NoPunctWER с моделями NeMo ASR.

    Выполняет две ключевые задачи:
    1. Инструментирует модель для перехвата данных WER
    2. Преобразует эти данные для использования метрикой NoPunctWER

    Поддерживает все форматы батчей NeMo, включая PromptedAudioToTextMiniBatch.
    """

    def __init__(self):
        super().__init__()
        self.no_punct_metric = None  # Унифицированное именование атрибута
        self.metric = None
        self._original_update = None
        self._original_compute = None

    def setup(self, trainer, pl_module, stage=None):
        # Инициализация метрики
        from metrics.no_punct_wer import NoPunctWER
        self.metric = NoPunctWER()

        # Инструментирование метрики WER модели
        if hasattr(pl_module, 'wer'):
            self._instrument_wer_metric(pl_module.wer)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Подготовка к валидации."""
        # Убеждаемся, что метрика инициализирована
        if self.no_punct_metric is None:
            from metrics.no_punct_wer import NoPunctWER
            self.no_punct_metric = NoPunctWER(dist_sync_on_step=False)
            self.no_punct_metric = self.no_punct_metric.to(pl_module.device)

        # Сбрасываем состояние метрики
        self.no_punct_metric.reset()

        # Инструментируем WER метрику, если необходимо
        if not self._is_instrumented:
            self._instrument_wer_metric(pl_module)

    def _instrument_wer_metric(self, wer_metric):
        """
        Инструментирует оригинальную метрику WER для сбора данных
        без нарушения её основной функциональности.
        """
        # Сохраняем ссылку на оригинальный метод update
        if hasattr(wer_metric, 'update') and self._original_update is None:
            self._original_update = wer_metric.update

            # Создаем прокси-метод с корректной передачей аргументов
            def instrumented_update(predictions, predictions_lengths, targets, targets_lengths, *args, **kwargs):
                # Исключаем дублирование параметров при вызове оригинального метода
                # Фильтруем kwargs, чтобы исключить параметры, которые уже есть в сигнатуре
                filtered_kwargs = {k: v for k, v in kwargs.items()
                                   if k not in ['predictions', 'predictions_lengths', 'targets', 'targets_lengths']}

                # Вызываем оригинальный метод с корректными аргументами
                result = self._original_update(
                    predictions,
                    predictions_lengths,
                    targets,
                    targets_lengths,
                    *args,
                    **filtered_kwargs
                )

                try:
                    # Преобразуем прогнозы и эталоны в текст
                    hyps, refs = self._extract_text_data(
                        wer_metric,
                        predictions,
                        predictions_lengths,
                        targets,
                        targets_lengths
                    )

                    # Обновляем нашу метрику без пунктуации
                    if hyps and refs:
                        self.metric.update(hyps, refs)
                except Exception as e:
                    # Логируем ошибку, но не прерываем основной процесс
                    import logging
                    logging.warning(f"Ошибка при обновлении метрики WER без пунктуации: {e}")

                return result

            # Заменяем метод update на инструментированную версию
            wer_metric.update = instrumented_update

    def _extract_data_from_wer(self, wer_metric) -> Tuple[List[str], List[str]]:
        """
        Извлекает данные из метрики WER.
        Адаптировано для различных реализаций WER в NeMo.
        """
        hyps, refs = [], []

        # Проверка различных атрибутов, где могут храниться данные
        for hyp_attr in ['_hypotheses', 'hypotheses', '_predictions', 'predictions']:
            if hasattr(wer_metric, hyp_attr) and getattr(wer_metric, hyp_attr):
                hyps = getattr(wer_metric, hyp_attr)
                break

        for ref_attr in ['_references', 'references', '_targets', 'targets']:
            if hasattr(wer_metric, ref_attr) and getattr(wer_metric, ref_attr):
                refs = getattr(wer_metric, ref_attr)
                break

        # Преобразование данных в строки
        hyps = [h.text if hasattr(h, 'text') else str(h) for h in hyps]
        refs = [str(r) for r in refs]

        return hyps, refs

    def _process_predictions_targets(self, wer_metric, predictions, targets) -> Tuple[List[str], List[str]]:
        """
        Преобразует предсказания и цели в текстовые строки.
        Использует декодер модели, если доступен.
        """
        hyps, refs = [], []

        # Использование декодера модели, если доступен
        if hasattr(wer_metric, 'decoding'):
            decoder = wer_metric.decoding

            # Обработка предсказаний
            if hasattr(decoder, 'ctc_decoder_predictions_tensor'):
                try:
                    if hasattr(predictions, 'shape') and len(predictions.shape) > 1:
                        # Данные в формате логитов/лог-вероятностей
                        lengths = torch.tensor([predictions.shape[1]] * predictions.shape[0])
                        batch_hyps = decoder.ctc_decoder_predictions_tensor(
                            predictions, lengths, return_hypotheses=True
                        )
                        hyps = [h.text if hasattr(h, 'text') else str(h) for h in batch_hyps]
                except Exception as e:
                    logging.debug(f"Не удалось декодировать предсказания: {str(e)}")

            # Обработка эталонов
            if hasattr(decoder, 'decode_tokens_to_str'):
                try:
                    if isinstance(targets, torch.Tensor) and targets.dim() > 1:
                        for i in range(targets.shape[0]):
                            tokens = targets[i].cpu().numpy().tolist()
                            refs.append(decoder.decode_tokens_to_str(tokens))
                except Exception as e:
                    logging.debug(f"Не удалось декодировать эталоны: {str(e)}")

        # Если данные уже в текстовом формате
        if not hyps and isinstance(predictions, list):
            hyps = [str(p) for p in predictions]

        if not refs and isinstance(targets, list):
            refs = [str(t) for t in targets]

        return hyps, refs

    def _extract_text_data(self, wer_metric, predictions, predictions_lengths, targets, targets_lengths):
        """
        Извлекает текстовые данные из тензоров прогнозов и эталонов.
        Использует декодер модели для корректного преобразования токенов в текст.
        """
        hypotheses, references = [], []

        # Используем декодер из метрики WER
        if hasattr(wer_metric, 'decoding'):
            decoder = wer_metric.decoding

            # Декодируем прогнозы
            if hasattr(decoder, 'ctc_decoder_predictions_tensor'):
                batch_hyps = decoder.ctc_decoder_predictions_tensor(
                    predictions, predictions_lengths, return_hypotheses=True
                )
                hypotheses = [h.text if hasattr(h, 'text') else str(h) for h in batch_hyps]

            # Декодируем эталоны
            if hasattr(decoder, 'decode_tokens_to_str'):
                for i in range(targets.size(0)):
                    target_len = targets_lengths[i].item()
                    target_tokens = targets[i][:target_len].cpu().numpy().tolist()
                    ref_text = decoder.decode_tokens_to_str(target_tokens)
                    references.append(ref_text)

        return hypotheses, references

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Вычисляет и логирует финальную метрику WER без пунктуации.
        """
        if self.metric is None or self.metric.total_words.item() == 0:
            import logging
            logging.warning("Недостаточно данных для расчёта WER без пунктуации")
            return

        # Вычисляем метрику
        wer_no_punct = self.metric.compute()

        # Логируем результат
        pl_module.log('val_wer_no_punct', wer_no_punct, prog_bar=True, sync_dist=True)

        # Добавляем в метрики для EarlyStopping
        trainer.callback_metrics["val_wer_no_punct"] = wer_no_punct

        # Вывод статистики
        self._print_metrics(wer_no_punct, pl_module)

        # Получение оригинального WER для сравнения
        original_wer = None
        if hasattr(pl_module, 'wer') and hasattr(pl_module.wer, 'compute'):
            try:
                original_wer_result = pl_module.wer.compute()
                if isinstance(original_wer_result, tuple):
                    original_wer = original_wer_result[0]
                else:
                    original_wer = original_wer_result

                if isinstance(original_wer, torch.Tensor):
                    original_wer = original_wer.item()
            except:
                pass

        self._print_detailed_report(wer_no_punct, original_wer, self.no_punct_metric.sample_count.item())

    def _print_metrics(self, wer_no_punct, pl_module):
        """Вывод базовой статистики в лог."""
        logging.info(f"WER без пунктуации: {wer_no_punct:.4f} ({wer_no_punct.item() * 100:.2f}%)")
        logging.info(f"Обработано примеров: {self.no_punct_metric.sample_count.item()}")
        logging.info(f"Всего слов: {self.no_punct_metric.word_count.item()}")

    def _print_detailed_report(self, wer_no_punct, original_wer, examples_count):
        """
        Форматирует и выводит детальный отчёт о метриках качества распознавания.
        Включает сравнение стандартного WER и WER без пунктуации.
        """
        import math

        def format_wer(wer_value):
            """Форматирует значение WER с интерпретацией качества."""
            if wer_value is None:
                return "N/A (недоступно)"

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

        # Расчёт улучшения (если доступны оба значения)
        improvement = "N/A"
        if original_wer is not None and isinstance(wer_no_punct, torch.Tensor):
            wer_np_value = wer_no_punct.item()
            if original_wer > 0 and not math.isinf(wer_np_value):
                rel_improvement = (original_wer - wer_np_value) / original_wer * 100
                improvement = f"{rel_improvement:.1f}%"

        # Формирование красивой таблицы
        logging.info("╔════════════════════════════════════════════════════════════╗")
        logging.info("║             МЕТРИКИ КАЧЕСТВА РАСПОЗНАВАНИЯ                 ║")
        logging.info("╠════════════════════════════════════════════════════════════╣")

        if original_wer is not None:
            logging.info(f"║ WER с пунктуацией:  {format_wer(original_wer):<35}  ║")

        logging.info(f"║ WER без пунктуации: {format_wer(wer_no_punct):<35}  ║")

        if original_wer is not None:
            logging.info(f"║ Относительное улучшение: {improvement:<31}  ║")

        logging.info("╠════════════════════════════════════════════════════════════╣")
        logging.info(f"║ Обработано примеров: {examples_count:<34}  ║")
        logging.info(f"║ Всего слов в эталоне: {self.no_punct_metric.word_count.item():<32}  ║")
        logging.info(f"║ Всего ошибок: {self.no_punct_metric.error_sum.item():<39}  ║")
        logging.info("╚════════════════════════════════════════════════════════════╝")