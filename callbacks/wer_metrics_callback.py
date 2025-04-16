# callbacks/wer_metrics_callback.py
import lightning.pytorch as pl

from nemo.utils import logging

from metrics.no_punct_wer import NoPunctWER


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

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch,
                                batch_idx: int, dataloader_idx: int = 0):
        if hasattr(outputs, 'preds') and hasattr(batch, 'tokens'):
            preds = pl_module.decode(outputs['preds'])
            refs = [pl_module.tokenizer.ids_to_text(token_ids) for token_ids in batch.tokens]
            self.val_predictions.extend(preds)
            self.val_references.extend(refs)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.val_predictions and self.val_references:
            self.wer_metric.update(self.val_predictions, self.val_references)
            wer_value = self.wer_metric.compute()
            pl_module.log('val_wer_no_punct', wer_value, sync_dist=True)
            logging.info(f"Validation No Punctuation WER: {wer_value}")
            self.wer_metric.reset()
            self.val_predictions = []
            self.val_references = []

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int,
                          dataloader_idx: int = 0):
        if hasattr(outputs, 'preds') and hasattr(batch, 'tokens'):
            preds = pl_module.decode(outputs['preds'])
            refs = [pl_module.tokenizer.ids_to_text(token_ids) for token_ids in batch.tokens]
            self.test_predictions.extend(preds)
            self.test_references.extend(refs)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.test_predictions and self.test_references:
            self.wer_metric.update(self.test_predictions, self.test_references)
            wer_value = self.wer_metric.compute()
            pl_module.log('test_wer_no_punct', wer_value, sync_dist=True)
            logging.info(f"Test No Punctuation WER: {wer_value}")
            self.wer_metric.reset()
            self.test_predictions = []
            self.test_references = []
