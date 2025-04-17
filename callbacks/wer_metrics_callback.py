from metrics.no_punct_wer import NoPunctWER
import lightning.pytorch as pl
import math
from nemo.utils import logging
import torch
import torch.distributed as dist


class NoPunctWERCallback(pl.Callback):
    """
    Коллбэк Lightning для вычисления Word Error Rate без пунктуации.
    """
    def __init__(self):
        super().__init__()
        self.wer_metric = NoPunctWER(dist_sync_on_step=False)
        self.val_predictions = []
        self.val_references = []
        self.test_predictions = []
        self.test_references = []
        self._is_distributed = dist.is_initialized() if dist.is_available() else False

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

    def _gather_predictions_and_references(self, predictions, references):
        """Собирает предсказания и референсы со всех процессов"""
        if not self._is_distributed:
            return predictions, references

        # Собираем все предсказания и референсы
        gathered_predictions = [None for _ in range(dist.get_world_size())]
        gathered_references = [None for _ in range(dist.get_world_size())]
        
        dist.all_gather_object(gathered_predictions, predictions)
        dist.all_gather_object(gathered_references, references)
        
        # Объединяем результаты
        all_predictions = []
        all_references = []
        for preds, refs in zip(gathered_predictions, gathered_references):
            all_predictions.extend(preds)
            all_references.extend(refs)
            
        return all_predictions, all_references

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0):
        if hasattr(outputs, 'preds') and hasattr(batch, 'tokens'):
            preds = pl_module.decode(outputs['preds'])
            refs = [pl_module.tokenizer.ids_to_text(token_ids) for token_ids in batch.tokens]
            self.val_predictions.extend(preds)
            self.val_references.extend(refs)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.val_predictions and self.val_references:
            # Собираем данные со всех процессов
            all_predictions, all_references = self._gather_predictions_and_references(
                self.val_predictions, self.val_references
            )
            
            # Обновляем метрику только на главном процессе
            if not self._is_distributed or dist.get_rank() == 0:
                self.wer_metric.update(all_predictions, all_references)
                wer_no_punct_value = self.wer_metric.compute()
                
                # Попытка получить оригинальный WER, если он логируется моделью
                original_wer_value = pl_module.trainer.callback_metrics.get('val_wer', float('inf'))
                cleaned_references = [self.wer_metric._normalize_text(ref) for ref in all_references]

                # Красивый вывод в лог
                logging.info("╔════════════════════════════════════════════════════════════╗")
                logging.info("║             МЕТРИКИ КАЧЕСТВА РАСПОЗНАВАНИЯ                 ║")
                logging.info("╠════════════════════════════════════════════════════════════╣")
                logging.info(f"║ WER с пунктуацией:  {self.format_wer(original_wer_value):<35}  ║")
                logging.info(f"║ WER без пунктуации: {self.format_wer(wer_no_punct_value):<35}  ║")
                logging.info("╠════════════════════════════════════════════════════════════╣")
                logging.info(f"║ Обработано примеров:{len(cleaned_references):<35}  ║")
                logging.info("╚════════════════════════════════════════════════════════════╝")

            # Синхронизируем метрику между процессами
            if self._is_distributed:
                if dist.get_rank() == 0:
                    wer_tensor = torch.tensor(wer_no_punct_value, device=pl_module.device)
                else:
                    wer_tensor = torch.tensor(0.0, device=pl_module.device)
                dist.broadcast(wer_tensor, src=0)
                wer_no_punct_value = wer_tensor.item()

            pl_module.log('val_wer_no_punct', wer_no_punct_value, sync_dist=True)

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
            # Собираем данные со всех процессов
            all_predictions, all_references = self._gather_predictions_and_references(
                self.test_predictions, self.test_references
            )
            
            # Обновляем метрику только на главном процессе
            if not self._is_distributed or dist.get_rank() == 0:
                self.wer_metric.update(all_predictions, all_references)
                wer_no_punct_value = self.wer_metric.compute()
                
                # Аналогично для тестовой выборки
                original_wer_value = pl_module.trainer.callback_metrics.get('test_wer', float('inf'))
                cleaned_references = [self.wer_metric._normalize_text(ref) for ref in all_references]

                logging.info("╔════════════════════════════════════════════════════════════╗")
                logging.info("║             МЕТРИКИ КАЧЕСТВА РАСПОЗНАВАНИЯ (ТЕСТ)          ║")
                logging.info("╠════════════════════════════════════════════════════════════╣")
                logging.info(f"║ WER с пунктуацией:  {self.format_wer(original_wer_value):<35}  ║")
                logging.info(f"║ WER без пунктуации: {self.format_wer(wer_no_punct_value):<35}  ║")
                logging.info("╠════════════════════════════════════════════════════════════╣")
                logging.info(f"║ Обработано примеров:{len(cleaned_references):<35}  ║")
                logging.info("╚════════════════════════════════════════════════════════════╝")

            # Синхронизируем метрику между процессами
            if self._is_distributed:
                if dist.get_rank() == 0:
                    wer_tensor = torch.tensor(wer_no_punct_value, device=pl_module.device)
                else:
                    wer_tensor = torch.tensor(0.0, device=pl_module.device)
                dist.broadcast(wer_tensor, src=0)
                wer_no_punct_value = wer_tensor.item()

            pl_module.log('test_wer_no_punct', wer_no_punct_value, sync_dist=True)

            self.wer_metric.reset()
            self.test_predictions = []
            self.test_references = []