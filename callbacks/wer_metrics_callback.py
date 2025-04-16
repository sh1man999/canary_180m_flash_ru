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
        self.no_punct_metric = None
        self._original_compute = None
        self._original_update = None
        self._is_instrumented = False

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        """Настраивает колбэк при инициализации тренера."""
        # Инициализируем метрику
        from metrics.no_punct_wer import NoPunctWER
        self.no_punct_metric = NoPunctWER(dist_sync_on_step=False)

        # Переносим метрику на нужное устройство
        if hasattr(pl_module, 'device'):
            self.no_punct_metric = self.no_punct_metric.to(pl_module.device)

        # Инструментируем метрику WER модели
        self._instrument_wer_metric(pl_module)

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

    def _instrument_wer_metric(self, pl_module: pl.LightningModule) -> None:
        """
        Инструментирует метрику WER модели для перехвата данных.
        Сохраняет оригинальное поведение метрики.
        """
        # Проверка наличия метрики WER
        if not hasattr(pl_module, 'wer'):
            logging.warning("Модель не имеет атрибута 'wer'. Расчёт WER без пунктуации будет недоступен.")
            return

        wer_metric = pl_module.wer

        # Инструментирование метода compute
        if hasattr(wer_metric, 'compute') and self._original_compute is None:
            self._original_compute = wer_metric.compute

            def instrumented_compute(*args, **kwargs):
                # Вызов оригинального метода
                result = self._original_compute(*args, **kwargs)

                try:
                    # Извлечение накопленных текстов и гипотез
                    hyps, refs = self._extract_data_from_wer(wer_metric)

                    # Обновление нашей метрики, если есть данные
                    if hyps and refs:
                        self.no_punct_metric.update(hyps, refs)
                        logging.debug(f"Метрика WER без пунктуации обновлена {len(hyps)} примерами")
                except Exception as e:
                    logging.warning(f"Не удалось обновить метрику WER без пунктуации: {str(e)}")

                return result

            # Замена метода
            wer_metric.compute = instrumented_compute

        # Инструментирование метода update для моделей с другим API
        if hasattr(wer_metric, 'update') and self._original_update is None:
            self._original_update = wer_metric.update

            def instrumented_update(predictions, targets, *args, **kwargs):
                # Вызов оригинального метода
                result = self._original_update(predictions, targets, *args, **kwargs)

                try:
                    # Преобразование предсказаний/целей в тексты
                    hyps, refs = self._process_predictions_targets(wer_metric, predictions, targets)

                    # Обновление нашей метрики
                    if hyps and refs:
                        self.no_punct_metric.update(hyps, refs)
                except Exception as e:
                    logging.debug(f"Не удалось обработать данные в методе update: {str(e)}")

                return result

            # Замена метода
            wer_metric.update = instrumented_update

        self._is_instrumented = True
        logging.info("Метрика WER успешно инструментирована для расчёта WER без пунктуации")

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

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Вычисляет и логирует метрику WER без пунктуации.
        """
        if self.no_punct_metric is None or self.no_punct_metric.word_count.item() == 0:
            logging.warning("Недостаточно данных для расчёта WER без пунктуации")
            return

        # Вычисление метрики
        wer_no_punct = self.no_punct_metric.compute()

        # Логирование
        pl_module.log('val_wer_no_punct', wer_no_punct, prog_bar=True, sync_dist=True)

        # Добавление в callback_metrics для EarlyStopping
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