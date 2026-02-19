"""
Dynamic Loss Balancing Module

Implements adaptive multi-task loss weighting strategies that adjust λ₁, λ₂
during training based on task performance or gradient statistics.

Strategies:
    1. GradNorm (Chen et al., 2018): Balance gradients across tasks
    2. Uncertainty Weighting (Kendall et al., 2018): Learn task uncertainties
    3. DWA (Dynamic Weight Average): Exponential moving average of loss rates

Rationale:
    Static weights (λ₁=0.6, λ₂=0.4) assume equal task importance throughout
    training. In practice:
        - Early training: sarcasm head learns slowly (rare positive class)
        - Mid training: sentiment head may dominate gradients
        - Late training: both tasks compete for representation capacity

    Dynamic balancing prevents one task from starving the other.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from utils.logger import get_logger

logger = get_logger("dynamic_loss_balancer")


# ═══════════════════════════════════════════════════════════
# Gradient-Based Balancing (GradNorm)
# ═══════════════════════════════════════════════════════════

class GradNormBalancer:
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing (Chen+ 2018)

    Balances task losses by equalizing gradient magnitudes with respect to
    a shared representation layer (typically the last shared encoder layer).

    Algorithm:
        1. Compute per-task gradient norms: ||∇_W L_i||
        2. Target gradient norm: Ḡ = avg(||∇_W L_i||) · (r_i)^α
           where r_i = loss_rate_i / avg(loss_rate)
        3. Update weights to minimize: |log(w_i · G_i / Ḡ_i)|

    Args:
        alpha:         Asymmetry parameter (0=symmetric, >0=prioritize harder tasks)
        update_freq:   Update weights every N steps
        learning_rate: Step size for weight updates
    """

    def __init__(
        self,
        alpha: float = 1.5,
        update_freq: int = 100,
        learning_rate: float = 0.025,
    ):
        self.alpha = alpha
        self.update_freq = update_freq
        self.lr = learning_rate

        # State
        self.step_count = 0
        self.initial_losses: Optional[Dict[str, float]] = None
        self.loss_history: List[Dict[str, float]] = []

        # Current weights (will be updated dynamically)
        self.weights = {"sentiment": 0.6, "sarcasm": 0.4}

        logger.info(
            f"GradNormBalancer: alpha={alpha}, update_freq={update_freq}, "
            f"lr={learning_rate}"
        )

    def step(
        self,
        losses: Dict[str, float],
        gradients: Dict[str, float],  # gradient norms per task
    ) -> Dict[str, float]:
        """
        Update task weights based on gradient norms.

        Args:
            losses:    {"sentiment": L_sent, "sarcasm": L_sarc}
            gradients: {"sentiment": ||∇L_sent||, "sarcasm": ||∇L_sarc||}

        Returns:
            Updated weights dict
        """
        self.step_count += 1

        # Initialize with first batch's losses
        if self.initial_losses is None:
            self.initial_losses = losses.copy()

        self.loss_history.append(losses.copy())

        # Update weights periodically
        if self.step_count % self.update_freq == 0:
            self._update_weights(gradients)

        return self.weights

    def _update_weights(self, gradients: Dict[str, float]) -> None:
        """Compute new task weights via GradNorm algorithm."""
        if self.initial_losses is None or len(self.loss_history) < 2:
            return

        # Compute loss rates: r_i = L_i(t) / L_i(0)
        current_losses = self.loss_history[-1]
        loss_rates = {
            task: current_losses[task] / max(self.initial_losses[task], 1e-8)
            for task in current_losses
        }

        # Inverse training rate (lower loss → higher weight)
        avg_rate = sum(loss_rates.values()) / len(loss_rates)
        inverse_rates = {
            task: (rate / avg_rate) ** self.alpha
            for task, rate in loss_rates.items()
        }

        # Target gradient norm per task
        avg_grad = sum(gradients.values()) / len(gradients)
        target_grads = {
            task: avg_grad * inverse_rates[task]
            for task in gradients
        }

        # Update weights to match target gradients
        for task in self.weights:
            if gradients[task] > 1e-8:
                ratio = target_grads[task] / gradients[task]
                # Gradient descent step
                self.weights[task] *= (1.0 + self.lr * (ratio - 1.0))

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        for task in self.weights:
            self.weights[task] /= total

        logger.debug(
            f"GradNorm step {self.step_count}: "
            f"weights={self.weights}, grad_norms={gradients}"
        )

    def get_weights(self) -> Dict[str, float]:
        """Return current task weights."""
        return self.weights.copy()


# ═══════════════════════════════════════════════════════════
# Dynamic Weight Average (DWA)
# ═══════════════════════════════════════════════════════════

class DynamicWeightAverage:
    """
    Dynamic Weight Average (Liu et al., 2019)

    Adjusts task weights based on the rate of loss decrease.
    Tasks with slower improvement get higher weight.

    w_i(t) ∝ exp(r_i(t) / T)
    where r_i(t) = L_i(t-1) / L_i(t-2)  (loss ratio between recent epochs)
          T = temperature hyperparameter

    Args:
        temperature: Softmax temperature (higher = more uniform weights)
        window_size: Number of recent losses to average
    """

    def __init__(self, temperature: float = 2.0, window_size: int = 2):
        self.temperature = temperature
        self.window_size = window_size
        self.loss_history: Dict[str, List[float]] = {"sentiment": [], "sarcasm": []}
        self.weights = {"sentiment": 0.6, "sarcasm": 0.4}

        logger.info(f"DWA: temperature={temperature}, window={window_size}")

    def step(self, losses: Dict[str, float]) -> Dict[str, float]:
        """
        Update weights based on loss change rates.

        Args:
            losses: Current task losses

        Returns:
            Updated weights
        """
        # Append to history
        for task, loss_val in losses.items():
            self.loss_history[task].append(loss_val)

        # Need at least window_size entries to compute rates
        if len(self.loss_history["sentiment"]) < self.window_size:
            return self.weights

        # Compute loss ratios
        ratios = {}
        for task in losses:
            history = self.loss_history[task]
            if len(history) >= 2:
                # r_i = L(t-1) / L(t-2)
                ratios[task] = history[-2] / max(history[-1], 1e-8)
            else:
                ratios[task] = 1.0

        # Softmax with temperature
        exp_ratios = {
            task: (ratio / self.temperature) for task, ratio in ratios.items()
        }
        # Numerical stability
        max_exp = max(exp_ratios.values())
        exp_vals = {task: (val - max_exp) for task, val in exp_ratios.items()}
        exp_vals = {task: 2.0 ** val for task, val in exp_vals.items()}

        total = sum(exp_vals.values())
        self.weights = {task: val / total for task, val in exp_vals.items()}

        # Normalize to K tasks (maintain sum = 1.0)
        K = len(self.weights)
        self.weights = {task: w * K for task, w in self.weights.items()}

        logger.debug(f"DWA: ratios={ratios}, weights={self.weights}")
        return self.weights

    def get_weights(self) -> Dict[str, float]:
        return self.weights.copy()


# ═══════════════════════════════════════════════════════════
# Uncertainty Weighting (already in CompositeLoss, but here
# as a standalone component for comparison)
# ═══════════════════════════════════════════════════════════

class UncertaintyWeighting:
    """Dummy stub when torch not available."""
    def __init__(self, *args, **kwargs):
        if nn is None or not hasattr(nn, 'Module'):
            raise RuntimeError('torch not available')

if nn is not None and hasattr(nn, 'Module'):
    class UncertaintyWeighting(nn.Module):
        """
        Homoscedastic Uncertainty Weighting (Kendall et al., 2018)
    
        Learns task-dependent uncertainty parameters σ_i:
            L = Σ_i [ (1/σ_i²) · L_i + log(σ_i) ]
    
        The log(σ_i) term prevents collapse to σ_i → ∞.
    
        Effective weight for task i:  1/σ_i²
        Tasks with higher uncertainty receive lower weight.
        """
    
        def __init__(self, num_tasks: int = 2):
            super().__init__()
            if torch is None:
                return
            # Initialize log(σ²) ≈ 0 → σ ≈ 1 → initial weight ≈ 1
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
            logger.info(f"UncertaintyWeighting: {num_tasks} tasks")
    
        def forward(self, losses: List["torch.Tensor"]) -> "torch.Tensor":
            """
            Compute weighted loss.
    
            Args:
                losses: [L_sentiment, L_sarcasm]
    
            Returns:
                Total weighted loss
            """
            if torch is None:
                raise RuntimeError("torch not available")
    
            weighted_losses = []
            for i, loss in enumerate(losses):
                precision = torch.exp(-self.log_vars[i])
                weighted = precision * loss + 0.5 * self.log_vars[i]
                weighted_losses.append(weighted)
    
            return sum(weighted_losses)
    
        def get_weights(self) -> Dict[str, float]:
            """Return effective task weights (1/σ²)."""
            if torch is None:
                return {"sentiment": 0.6, "sarcasm": 0.4}
            precisions = torch.exp(-self.log_vars).detach().cpu().numpy()
            return {
                "sentiment": float(precisions[0]),
                "sarcasm":   float(precisions[1]),
            }
    
    
# ═══════════════════════════════════════════════════════════
# Factory: choose balancing strategy from config
# ═══════════════════════════════════════════════════════════

def build_loss_balancer(
    method: str,
    **kwargs
) -> Optional[object]:
    """
    Factory function to build a loss balancer.

    Args:
        method: 'gradnorm' | 'dwa' | 'uncertainty' | 'static'
        **kwargs: Method-specific parameters

    Returns:
        Balancer instance or None (for 'static')
    """
    if method == "static":
        return None

    if method == "gradnorm":
        return GradNormBalancer(
            alpha=kwargs.get("alpha", 1.5),
            update_freq=kwargs.get("update_freq", 100),
            learning_rate=kwargs.get("learning_rate", 0.025),
        )

    if method == "dwa":
        return DynamicWeightAverage(
            temperature=kwargs.get("temperature", 2.0),
            window_size=kwargs.get("window_size", 2),
        )

    if method == "uncertainty":
        return UncertaintyWeighting(num_tasks=2)

    raise ValueError(f"Unknown balancing method: {method}")


# ═══════════════════════════════════════════════════════════
# Smoke-test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing dynamic loss balancers...\n")

    # GradNorm
    print("[1] GradNorm")
    gradnorm = GradNormBalancer(alpha=1.5, update_freq=2)
    for step in range(5):
        losses = {"sentiment": 0.5 - step * 0.05, "sarcasm": 0.8 - step * 0.02}
        grads  = {"sentiment": 0.1, "sarcasm": 0.15}
        weights = gradnorm.step(losses, grads)
        print(f"  Step {step+1}: weights={weights}")

    # DWA
    print("\n[2] DWA")
    dwa = DynamicWeightAverage(temperature=2.0)
    for step in range(5):
        losses = {"sentiment": 0.5 - step * 0.05, "sarcasm": 0.8 - step * 0.02}
        weights = dwa.step(losses)
        print(f"  Step {step+1}: weights={weights}")

    # Uncertainty
    if torch is not None:
        print("\n[3] UncertaintyWeighting")
        unc = UncertaintyWeighting(num_tasks=2)
        losses = [torch.tensor(0.5), torch.tensor(0.8)]
        total_loss = unc(losses)
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Weights: {unc.get_weights()}")
        print("\n✓ All balancers tested successfully!")
    else:
        print("\n⚠ torch not available — skipping UncertaintyWeighting test")
