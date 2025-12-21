"""
Bayesian Posterior Distributions for Adaptive Branching MCTS.

This module implements conjugate belief distributions for value estimation.
We support two primary distributions:
1. NormalInverseGammaPosterior: For unbounded rewards (approx. Gaussian).
   - Reference: Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. Section 4.6.3.
2. BetaPosterior: For bounded [0, 1] rewards (Bernoulli/Beta).

Mathematical Rigor:
-------------------
For a likelihood x ~ N(mu, sigma^2), the conjugate prior is NIG(mu, sigma^2 | m, kappa, alpha, beta).
The marginal posterior for mu is a Student-t distribution, but for Thompson Sampling
we sample the joint posterior (mu, sigma^2) directly:
    1. sigma^2 ~ InvGamma(alpha, beta)
    2. mu ~ N(mean, sigma^2 / kappa)

"""

import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random

# Use standard library random for sampling to avoid Heavy numpy dependency in runtime
# But for Gamma/InvGamma we might need math.gamma or robust approximation.
# Or simpler: use random.gammavariate

logger = logging.getLogger(__name__)


class ProbabilisticPosterior(ABC):
    """Abstract base class for Bayesian posteriors."""

    @abstractmethod
    def update(self, value: float) -> None:
        """Update belief with a new observation."""
        pass

    @abstractmethod
    def sample(self) -> float:
        """Draw a sample from the posterior predictive (or belief) distribution."""
        pass

    @property
    @abstractmethod
    def n_observations(self) -> int:
        pass


@dataclass
class NormalInverseGammaPosterior(ProbabilisticPosterior):
    """
    Normal-Inverse-Gamma (NIG) posterior for unknown mean and variance.

    Parameters:
        mu (float): Prior estimate of the mean.
        kappa (float): Confidence in the prior mean (pseudo-observations).
        alpha (float): Shape parameter for the variance (InvGamma).
        beta (float): Scale parameter for the variance (InvGamma).

    Defaults (Weakly informative):
        mu=0.0, kappa=1.0, alpha=2.0, beta=1.0
        Note: alpha > 1 is required for mean variance existence.
    """

    mu: float = 0.0
    kappa: float = 1.0
    alpha: float = 2.0
    beta: float = 1.0

    _n: int = field(default=0, init=False, repr=False)

    def update(self, x: float) -> None:
        """
        Incremental update for a single observation x.
        Formulas derived from Murphy (2012) for n=1.
        
        mu_n = (kappa * mu + x) / (kappa + 1)
        kappa_n = kappa + 1
        alpha_n = alpha + 0.5
        beta_n = beta + (kappa * (x - mu)**2) / (2 * (kappa + 1))
        """
        # Save current params for calculation
        mu_0 = self.mu
        kappa_0 = self.kappa
        # alpha_0 = self.alpha
        beta_0 = self.beta

        # Calculate update term for beta
        # Term: (n * kappa_0 / (kappa_0 + n)) * (mean_diff)**2 / 2
        # For n=1: (kappa_0 / (kappa_0 + 1)) * (x - mu_0)**2 / 2
        beta_update = (kappa_0 * (x - mu_0)**2) / (2.0 * (kappa_0 + 1.0))

        # Update parameters
        self.mu = (kappa_0 * mu_0 + x) / (kappa_0 + 1.0)
        self.kappa = kappa_0 + 1.0
        self.alpha = self.alpha + 0.5
        self.beta = beta_0 + beta_update

        self._n += 1

    def sample(self) -> float:
        """
        Thompson Sampling via the Joint Posterior.

        1. Sample variance sigma^2 ~ InvGamma(alpha, beta)
           Note: If X ~ Gamma(alpha, beta_rate), then 1/X ~ InvGamma(alpha, beta_rate).
           random.gammavariate(alpha, beta) uses 1/beta as scale? 
           Python docs: random.gammavariate(alpha, beta) -> PDF x^(alpha-1) * exp(-x/beta).
           This is Gamma(k=alpha, theta=beta).
           Standard Stats Gamma is G(alpha, rate). 
           So we need to be careful with parameterization.
           
           Target: InvGamma(alpha, beta).
           Let Y ~ Gamma(alpha, rate=beta). Then 1/Y ~ InvGamma(alpha, beta).
           Python gammavariate(alpha, 1.0) * beta is Gamma(alpha, theta=beta)? No.
           
           Let's rely on the definition:
           Standard formulation: Beta is the RATE parm for Gamma? Or scale?
           For NIG, beta is typically a sum-of-squares (scale-like).
           
           We use the relation:
           sigma_sq = 1.0 / random.gammavariate(self.alpha, 1.0 / self.beta)
           Here second arg to gammavariate is theta (scale) = 1/beta (if beta is rate).
           Wait, NIG beta is usually "rate" of the gamma? No, Murphy uses beta as rate?
           Murphy Eq 4.126: InvGamma(sigma^2 | alpha, beta).
           
           Let's assume standard InvGamma(alpha, beta) where mean = beta / (alpha - 1).
           Sampling: Y ~ Gamma(shape=alpha, scale=1/beta).
        """
        # Precision (1/sigma^2) ~ Gamma(alpha, rate=beta)
        # Using python's random.gammavariate(alpha, theta=1/beta)
        precision = random.gammavariate(self.alpha, 1.0 / self.beta)
        
        if precision <= 1e-9:
            precision = 1e-9  # Numerical guard
            
        variance = 1.0 / precision
        
        # 2. Sample mean mu ~ N(mu_n, variance / kappa_n)
        sigma_mean = math.sqrt(variance / self.kappa)
        sampled_mu = random.gauss(self.mu, sigma_mean)
        
        return sampled_mu

    @property
    def n_observations(self) -> int:
        return self._n


@dataclass
class BetaPosterior(ProbabilisticPosterior):
    """
    Beta posterior for bounded rewards in [0, 1].
    Conjugate for Bernoulli/Binomial likelihoods.
    
    Prior: Beta(alpha, beta).
    """
    alpha: float = 1.0
    beta: float = 1.0
    _n: int = field(default=0, init=False, repr=False)

    def update(self, value: float) -> None:
        """
        Update rule for Beta-Bernoulli.
        alpha_n = alpha + x
        beta_n = beta + (1 - x)
        
        (Generalized for x in [0, 1] as quasi-Bernoulli update)
        """
        # Clamp for safety
        val = max(0.0, min(1.0, value))
        self.alpha += val
        self.beta += (1.0 - val)
        self._n += 1

    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)

    @property
    def n_observations(self) -> int:
        return self._n
