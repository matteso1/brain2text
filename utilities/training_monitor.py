"""
Comprehensive training monitoring utilities for detecting and logging training issues.
Tracks gradients, predictions, memory, and detects anomalies in real-time.
"""
import torch
import numpy as np
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class TrainingMonitor:
    """
    Monitors training metrics and detects anomalies.

    Tracks:
    - Gradient norms per layer
    - Prediction diversity
    - Weight statistics
    - GPU memory usage
    - Training anomalies (explosions, NaN, mode collapse)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        log_dir: str,
        log_every_n_batches: int = 50,
        csv_log_every_n_batches: int = 200,
        grad_norm_threshold: float = 20.0,
        loss_spike_threshold: float = 3.0,
        min_unique_phonemes: int = 5
    ):
        self.model = model
        self.log_dir = log_dir
        self.log_every_n = log_every_n_batches
        self.csv_log_every_n = csv_log_every_n_batches

        # Thresholds for anomaly detection
        self.grad_norm_threshold = grad_norm_threshold
        self.loss_spike_threshold = loss_spike_threshold
        self.min_unique_phonemes = min_unique_phonemes

        # CSV files
        self.metrics_csv = os.path.join(log_dir, 'training_metrics.csv')
        self.gradients_csv = os.path.join(log_dir, 'layer_gradients.csv')
        self.validation_csv = os.path.join(log_dir, 'validation_metrics.csv')

        # Initialize CSV files
        self._init_csv_files()

        # Track history for anomaly detection
        self.loss_history = []
        self.grad_norm_history = []

    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        # Training metrics
        if not os.path.exists(self.metrics_csv):
            with open(self.metrics_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'step', 'loss', 'grad_norm', 'lr', 'gpu_mem_gb',
                    'unique_phonemes', 'blank_percent', 'diversity_score',
                    'max_grad_layer', 'warning_flags'
                ])

        # Layer gradients
        if not os.path.exists(self.gradients_csv):
            with open(self.gradients_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'step', 'input', 'gru1', 'gru2', 'gru3', 'gru4', 'gru5',
                    'output', 'day'
                ])

        # Validation metrics
        if not os.path.exists(self.validation_csv):
            with open(self.validation_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'avg_per', 'unique_phonemes_pred'])

    def compute_layer_grad_norms(self) -> Dict[str, float]:
        """Compute gradient norm for each layer in the model"""
        layer_grads = {}

        # Day layers
        day_grad_norms = []
        for i, param in enumerate(self.model.day_weights.parameters()):
            if param.grad is not None:
                day_grad_norms.append(param.grad.norm().item())
        for i, param in enumerate(self.model.day_biases.parameters()):
            if param.grad is not None:
                day_grad_norms.append(param.grad.norm().item())
        layer_grads['day'] = np.mean(day_grad_norms) if day_grad_norms else 0.0

        # GRU layers
        for i in range(self.model.n_layers):
            grad_norms = []
            for name, param in self.model.gru.named_parameters():
                if param.grad is not None and f'weight_ih_l{i}' in name or f'weight_hh_l{i}' in name:
                    grad_norms.append(param.grad.norm().item())
            layer_grads[f'gru{i+1}'] = np.mean(grad_norms) if grad_norms else 0.0

        # Output layer
        if self.model.out.weight.grad is not None:
            layer_grads['output'] = self.model.out.weight.grad.norm().item()
        else:
            layer_grads['output'] = 0.0

        # Input layer (day layer dropout)
        layer_grads['input'] = layer_grads['day']  # Approximate

        return layer_grads

    def analyze_predictions(self, logits: torch.Tensor) -> Dict[str, any]:
        """
        Analyze prediction diversity and distribution.

        Args:
            logits: (B, T, C) tensor of logits

        Returns:
            dict with unique_phonemes, blank_percent, top5_phonemes, diversity_score
        """
        # Greedy decode
        preds = logits.argmax(dim=-1).cpu().numpy()  # (B, T)

        # Flatten all predictions
        all_preds = preds.flatten()

        # Count unique phonemes
        unique_phonemes = len(np.unique(all_preds))

        # Blank percentage (phoneme 0)
        blank_percent = (all_preds == 0).sum() / len(all_preds) * 100

        # Top 5 most common phonemes
        phoneme_counts = np.bincount(all_preds, minlength=41)
        top5_phonemes = np.argsort(phoneme_counts)[-5:][::-1].tolist()

        # Diversity score (entropy of distribution)
        probs = phoneme_counts / phoneme_counts.sum()
        probs = probs[probs > 0]  # Remove zeros
        diversity_score = -np.sum(probs * np.log(probs + 1e-10)) / np.log(41)

        return {
            'unique_phonemes': unique_phonemes,
            'blank_percent': blank_percent,
            'top5_phonemes': top5_phonemes,
            'diversity_score': diversity_score
        }

    def get_gpu_memory(self) -> float:
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0

    def compute_total_grad_norm(self) -> float:
        """Compute total gradient norm across all parameters"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        return total_norm ** 0.5

    def check_anomalies(
        self,
        metrics: Dict[str, any],
        prev_loss: Optional[float] = None
    ) -> List[str]:
        """
        Check for training anomalies and return list of warnings.

        Args:
            metrics: Current metrics dict
            prev_loss: Previous loss value for spike detection

        Returns:
            List of warning strings
        """
        warnings = []

        # Gradient explosion
        if metrics['grad_norm'] > self.grad_norm_threshold:
            warnings.append(
                f"[ERROR] GRADIENT EXPLOSION! Grad norm {metrics['grad_norm']:.1f} "
                f"(threshold {self.grad_norm_threshold})"
            )

        # Loss spike
        if prev_loss is not None and metrics['loss'] > prev_loss * self.loss_spike_threshold:
            spike_ratio = metrics['loss'] / prev_loss
            warnings.append(
                f"[ERROR] LOSS SPIKE! {prev_loss:.2f} -> {metrics['loss']:.2f} "
                f"({spike_ratio:.1f}x increase)"
            )

        # Low diversity (mode collapse)
        if metrics['unique_phonemes'] < self.min_unique_phonemes:
            warnings.append(
                f"[ERROR] LOW DIVERSITY! Only {metrics['unique_phonemes']} unique phonemes "
                f"(mostly: {metrics['top5_phonemes'][0]})"
            )

        # Check per-layer gradient explosions
        max_layer = max(metrics['layer_grads'].items(), key=lambda x: x[1])
        if max_layer[1] > self.grad_norm_threshold:
            warnings.append(
                f"[ERROR] Layer '{max_layer[0]}' gradient exploded: {max_layer[1]:.1f}"
            )

        # High memory usage
        if metrics['gpu_mem_gb'] > 14.0:  # 16GB card, warning at 87.5%
            warnings.append(
                f"[WARNING] High memory usage: {metrics['gpu_mem_gb']:.1f}GB / 16GB"
            )

        # NaN check
        if np.isnan(metrics['loss']) or np.isinf(metrics['loss']):
            warnings.append("[ERROR] NaN/Inf DETECTED in loss!")

        return warnings

    def compute_metrics(
        self,
        loss: float,
        logits: torch.Tensor,
        step: int,
        lr: float
    ) -> Dict[str, any]:
        """
        Compute all metrics for current batch.

        Args:
            loss: Current loss value
            logits: Model output logits
            step: Current training step
            lr: Current learning rate

        Returns:
            Dict with all metrics
        """
        metrics = {
            'step': step,
            'loss': loss,
            'lr': lr,
            'grad_norm': self.compute_total_grad_norm(),
            'gpu_mem_gb': self.get_gpu_memory(),
            'layer_grads': self.compute_layer_grad_norms(),
        }

        # Prediction analysis
        pred_metrics = self.analyze_predictions(logits)
        metrics.update(pred_metrics)

        # Track history
        self.loss_history.append(loss)
        self.grad_norm_history.append(metrics['grad_norm'])

        # Keep only last 100 for anomaly detection
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)
            self.grad_norm_history.pop(0)

        return metrics

    def save_to_csv(self, metrics: Dict[str, any], warnings: List[str]):
        """Save metrics to CSV files"""
        # Training metrics
        with open(self.metrics_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics['step'],
                f"{metrics['loss']:.4f}",
                f"{metrics['grad_norm']:.2f}",
                f"{metrics['lr']:.6f}",
                f"{metrics['gpu_mem_gb']:.2f}",
                metrics['unique_phonemes'],
                f"{metrics['blank_percent']:.1f}",
                f"{metrics['diversity_score']:.3f}",
                max(metrics['layer_grads'].items(), key=lambda x: x[1])[0],
                '|'.join(warnings) if warnings else 'none'
            ])

        # Layer gradients
        with open(self.gradients_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            lg = metrics['layer_grads']
            writer.writerow([
                metrics['step'],
                f"{lg.get('input', 0):.2f}",
                f"{lg.get('gru1', 0):.2f}",
                f"{lg.get('gru2', 0):.2f}",
                f"{lg.get('gru3', 0):.2f}",
                f"{lg.get('gru4', 0):.2f}",
                f"{lg.get('gru5', 0):.2f}",
                f"{lg.get('output', 0):.2f}",
                f"{lg.get('day', 0):.2f}"
            ])

    def format_layer_grads(self, layer_grads: Dict[str, float]) -> str:
        """Format layer gradients for printing"""
        parts = []
        for name in ['input', 'gru1', 'gru2', 'gru3', 'gru4', 'gru5', 'output', 'day']:
            val = layer_grads.get(name, 0)
            # Add warning flag if gradient is high
            flag = " [!]" if val > self.grad_norm_threshold else ""
            parts.append(f"{name}={val:.1f}{flag}")
        return ' '.join(parts)

    def print_metrics(
        self,
        metrics: Dict[str, any],
        warnings: List[str],
        verbose: bool = True
    ):
        """
        Print metrics to console.

        Args:
            metrics: Metrics dict
            warnings: List of warning strings
            verbose: If True, print detailed stats
        """
        # Main line
        print(f"\nStep {metrics['step']} | "
              f"Loss: {metrics['loss']:.2f} | "
              f"Grad: {metrics['grad_norm']:.1f} | "
              f"Mem: {metrics['gpu_mem_gb']:.1f}GB | "
              f"Phonemes: {metrics['unique_phonemes']}/41")

        if verbose:
            # Layer gradients
            print(f"  Layer grads: {self.format_layer_grads(metrics['layer_grads'])}")

            # Predictions
            print(f"  Predictions: blank={metrics['blank_percent']:.0f}% "
                  f"top5={metrics['top5_phonemes']} "
                  f"unique={metrics['unique_phonemes']}")

        # Warnings in RED
        if warnings:
            for warning in warnings:
                print(f"\033[91m  {warning}\033[0m")  # Red text


def format_prediction_sample(pred_ids: List[int], true_ids: List[int], per: float) -> str:
    """Format a single prediction sample for display"""
    pred_str = str(pred_ids[:10]) + ('...' if len(pred_ids) > 10 else '')
    true_str = str(true_ids[:10]) + ('...' if len(true_ids) > 10 else '')
    return f"Pred={pred_str} | True={true_str} | PER={per:.0f}%"
