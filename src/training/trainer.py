"""
Trainer Module

Production-grade training loop for the multi-task sentiment model.

Features:
    - Gradient accumulation (effective batch size scaling)
    - Gradient clipping (prevents exploding gradients)
    - Linear warmup + linear decay schedule
    - Early stopping with configurable patience
    - Best-model checkpointing (by combined F1)
    - Per-epoch metrics logging (sentiment F1 + sarcasm F1)
    - Device-agnostic (CPU / CUDA / MPS)
    - Reproducibility via seed setting

Composite metric for early stopping:
    combined_f1 = 0.6 * macro_f1_sentiment + 0.4 * f1_sarcasm
    (mirrors the λ₁/λ₂ loss weights)
"""

import os
import time
import random
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader
except ImportError:
    torch = None  # type: ignore
    class _NNStub:
        class Module: pass
        def utils(self): pass
    nn = _NNStub()  # type: ignore

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger, MetricsLogger
from training.focal_loss import CompositeLoss
from training.train_utils import EarlyStopping, compute_epoch_metrics

logger = get_logger("trainer")


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CUDA ops (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# Learning Rate Scheduler
# ─────────────────────────────────────────────

def build_linear_warmup_scheduler(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Linear warmup for `num_warmup_steps`, then linear decay to 0.

    Standard schedule for fine-tuning Transformers (HuggingFace default).
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)



# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class Trainer:
    """
    Multi-task training orchestrator.

    Usage:
        trainer = Trainer(model, loss_fn, train_loader, val_loader, config)
        results = trainer.train()

    Args:
        model:           MultiTaskSentimentModel instance
        loss_fn:         CompositeLoss instance
        train_loader:    Training DataLoader
        val_loader:      Validation DataLoader
        training_config: Loaded training_config.yaml dict
        checkpoint_dir:  Directory to save model checkpoints
        device:          'cpu' | 'cuda' | 'mps' | None (auto-detect)
    """

    def __init__(
        self,
        model:           nn.Module,
        loss_fn:         CompositeLoss,
        train_loader:    DataLoader,
        val_loader:      DataLoader,
        training_config: dict,
        checkpoint_dir:  str = "checkpoints",
        device:          Optional[str] = None,
    ):
        self.model    = model
        self.loss_fn  = loss_fn
        self.train_dl = train_loader
        self.val_dl   = val_loader
        self.cfg      = training_config["training"]
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        logger.info(f"Training device: {self.device}")

        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)

        # Seed
        set_seed(self.cfg.get("seed", 42))

        # Optimiser with differential learning rates
        self.optimizer = self._build_optimizer()

        # Scheduler
        steps_per_epoch = len(self.train_dl)
        grad_accum      = self.cfg.get("gradient_accumulation_steps", 1)
        updates_per_epoch = steps_per_epoch // grad_accum
        total_updates   = updates_per_epoch * self.cfg["num_epochs"]
        warmup_steps    = self.cfg.get("scheduler", {}).get("num_warmup_steps", 500)

        self.scheduler = build_linear_warmup_scheduler(
            self.optimizer,
            num_warmup_steps=min(warmup_steps, total_updates // 10),
            num_training_steps=total_updates,
        )
        self.grad_accum = grad_accum
        self.max_grad_norm = self.cfg.get("max_grad_norm", 1.0)

        # Early stopping
        es_cfg = self.cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 3),
            min_delta=es_cfg.get("min_delta", 1e-3),
            mode="max",
        ) if es_cfg.get("enabled", True) else None

        # Metrics logging
        self.metrics_logger = MetricsLogger("logs/training_metrics.csv")
        self.best_combined_f1 = 0.0
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "val_sentiment_f1": [], "val_sarcasm_f1": [], "val_combined_f1": []
        }

    # ── Optimizer ────────────────────────────────────────────────────────────

    def _build_optimizer(self) -> AdamW:
        """Build AdamW with differential learning rates per param group."""
        opt_cfg = self.cfg.get("optimizer", {})
        encoder_lr = opt_cfg.get("learning_rate", 2e-5)
        head_lr    = opt_cfg.get("learning_rate", 2e-5) * 5  # 5× for heads
        wd         = opt_cfg.get("weight_decay", 0.01)

        if hasattr(self.model, "get_optimizer_param_groups"):
            param_groups = self.model.get_optimizer_param_groups(
                encoder_lr=encoder_lr, head_lr=head_lr, weight_decay=wd
            )
        else:
            param_groups = self.model.parameters()

        return AdamW(
            param_groups,
            lr=encoder_lr,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=float(opt_cfg.get("eps", 1e-8)),
        )

    # ── Training step ────────────────────────────────────────────────────────

    def _train_step(self, batch: Dict) -> Tuple[float, Dict]:
        """
        Forward + backward pass for one batch.

        Returns:
            (loss_value, component_dict)
        """
        input_ids       = batch["input_ids"].to(self.device)
        attention_mask  = batch["attention_mask"].to(self.device)
        lid_token_ids   = batch.get("lid_token_ids")
        if lid_token_ids is not None:
            lid_token_ids = lid_token_ids.to(self.device)
        sentiment_labels = batch["sentiment_label"].to(self.device)
        sarcasm_labels   = batch["sarcasm_label"].to(self.device)

        outputs = self.model(input_ids, attention_mask, lid_token_ids)

        total_loss, components = self.loss_fn(
            outputs["sentiment_logits"],
            outputs["sarcasm_logit"],
            sentiment_labels,
            sarcasm_labels,
        )

        # Scale by accumulation steps for correct gradient magnitude
        (total_loss / self.grad_accum).backward()

        return total_loss.item(), components

    # ── Evaluation step ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_epoch(self) -> Tuple[float, Dict]:
        """
        Full validation pass.

        Returns:
            (avg_val_loss, metrics_dict)
        """
        self.model.eval()

        total_loss = 0.0
        sent_preds, sent_labels = [], []
        sarc_preds, sarc_labels = [], []

        for batch in self.val_dl:
            input_ids       = batch["input_ids"].to(self.device)
            attention_mask  = batch["attention_mask"].to(self.device)
            lid_token_ids   = batch.get("lid_token_ids")
            if lid_token_ids is not None:
                lid_token_ids = lid_token_ids.to(self.device)

            s_labels = batch["sentiment_label"].to(self.device)
            r_labels = batch["sarcasm_label"].to(self.device)

            outputs = self.model(input_ids, attention_mask, lid_token_ids)
            loss, _ = self.loss_fn(
                outputs["sentiment_logits"], outputs["sarcasm_logit"],
                s_labels, r_labels
            )
            total_loss += loss.item()

            # Collect predictions
            s_pred = outputs["sentiment_logits"].argmax(dim=-1)
            r_prob = torch.sigmoid(outputs["sarcasm_logit"].squeeze(-1))
            r_pred = (r_prob >= 0.5).long()

            sent_preds.extend(s_pred.cpu().tolist())
            sent_labels.extend(s_labels.cpu().tolist())
            sarc_preds.extend(r_pred.cpu().tolist())
            sarc_labels.extend(r_labels.cpu().tolist())

        avg_loss = total_loss / len(self.val_dl)
        metrics  = compute_epoch_metrics(
            sent_preds, sent_labels, sarc_preds, sarc_labels
        )
        return avg_loss, metrics

    # ── Checkpointing ────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, combined_f1: float) -> None:
        """Save model if it achieves a new best combined F1."""
        ckpt_path = self.ckpt_dir / f"checkpoint_epoch{epoch:02d}_f1{combined_f1:.4f}.pt"
        torch.save(
            {
                "epoch":        epoch,
                "combined_f1":  combined_f1,
                "model_state":  self.model.state_dict(),
                "optim_state":  self.optimizer.state_dict(),
                "sched_state":  self.scheduler.state_dict(),
            },
            ckpt_path,
        )

        # Also maintain a "best model" symlink-equivalent
        best_path = self.ckpt_dir / "best_model.pt"
        shutil.copy(ckpt_path, best_path)

        # Prune old checkpoints (keep best + last 2)
        all_ckpts = sorted(
            self.ckpt_dir.glob("checkpoint_epoch*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        save_limit = self.cfg.get("save_total_limit", 3)
        for old in all_ckpts[:-save_limit]:
            old.unlink(missing_ok=True)

        logger.info(f"Checkpoint saved: {ckpt_path.name} (F1={combined_f1:.4f})")

    # ── Main train loop ──────────────────────────────────────────────────────

    def train(self) -> Dict:
        """
        Run the full training loop.

        Returns:
            Training history dict with per-epoch metrics
        """
        num_epochs  = self.cfg["num_epochs"]
        grad_accum  = self.grad_accum

        logger.info(
            f"Starting training: {num_epochs} epochs | "
            f"batch={self.cfg['batch_size']} | "
            f"grad_accum={grad_accum} | "
            f"device={self.device}"
        )

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # ── Training epoch ──────────────────────────────────────────
            self.model.train()
            self.optimizer.zero_grad()

            running_loss    = 0.0
            steps_this_epoch = 0
            num_batches     = len(self.train_dl)

            for step, batch in enumerate(self.train_dl, start=1):
                loss_val, components = self._train_step(batch)
                running_loss += loss_val

                # Gradient accumulation update
                if step % grad_accum == 0 or step == num_batches:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    steps_this_epoch += 1

                    # Periodic step log
                    log_every = max(1, num_batches // 5)
                    if steps_this_epoch % log_every == 0:
                        lr_now = self.scheduler.get_last_lr()[0]
                        logger.info(
                            f"  Epoch {epoch}/{num_epochs} "
                            f"step {steps_this_epoch} | "
                            f"loss={loss_val:.4f} | lr={lr_now:.2e} | "
                            f"L_sent={components['loss_sentiment']:.4f} "
                            f"L_sarc={components['loss_sarcasm']:.4f}"
                        )

            avg_train_loss = running_loss / num_batches

            # ── Validation epoch ────────────────────────────────────────
            avg_val_loss, val_metrics = self._eval_epoch()

            epoch_time = time.time() - epoch_start

            # Log epoch summary
            logger.info(
                f"Epoch {epoch}/{num_epochs} complete ({epoch_time:.1f}s) | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f} | "
                f"sentiment_f1={val_metrics['sentiment_f1']:.4f} | "
                f"sarcasm_f1={val_metrics['sarcasm_f1']:.4f} | "
                f"combined_f1={val_metrics['combined_f1']:.4f}"
            )

            # Persist metrics
            self.metrics_logger.log_epoch_metrics(
                epoch=epoch, step=steps_this_epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                **val_metrics,
            )

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["val_sentiment_f1"].append(val_metrics["sentiment_f1"])
            self.history["val_sarcasm_f1"].append(val_metrics["sarcasm_f1"])
            self.history["val_combined_f1"].append(val_metrics["combined_f1"])

            # Checkpoint if improved
            combined_f1 = val_metrics["combined_f1"]
            if combined_f1 > self.best_combined_f1:
                self.best_combined_f1 = combined_f1
                self._save_checkpoint(epoch, combined_f1)

            # Early stopping
            if self.early_stopping and self.early_stopping.step(combined_f1):
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best combined F1: {self.early_stopping.best_score:.4f}"
                )
                break

        logger.info(
            f"Training complete. Best combined F1: {self.best_combined_f1:.4f}"
        )
        return self.history


# ─────────────────────────────────────────────
# Convenience builder
# ─────────────────────────────────────────────

def build_trainer(
    model,
    loss_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_config: dict,
    checkpoint_dir: str = "checkpoints",
    device: Optional[str] = None,
) -> Trainer:
    """
    Factory function that creates a Trainer from config dicts.
    Keeps model-building code out of the main script.
    """
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
