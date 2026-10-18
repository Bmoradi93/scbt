"""Linear Thompson sampling (Agrawal & Goyal 2013) over a growing arm set.

Each arm j keeps a Bayesian linear regression r ~ w_j^T x with Gaussian
prior N(mu0, sigma0^2 I). New arms get an optimistic prior mean so they are
tried (paper Sec. IV-E). ``CALL_LLM`` is an ordinary arm whose reward
already includes the -kappa cost, so "stop calling the LLM" is learned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

CALL_LLM = "__call_llm__"


@dataclass
class ArmPosterior:
    dim: int
    prior_mean: float
    prior_var: float = 1.0
    noise_var: float = 0.25
    # Sufficient statistics: precision A = I/prior_var + X^T X / noise_var, b = A mu
    A: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    n: int = 0
    reward_sum: float = 0.0

    def __post_init__(self):
        if self.A is None:
            self.A = np.eye(self.dim) / self.prior_var
            # Prior mean implemented as a bias on the last (constant) feature.
            mu0 = np.zeros(self.dim); mu0[-1] = self.prior_mean
            self.b = self.A @ mu0

    def mean(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        cov = np.linalg.inv(self.A)
        return rng.multivariate_normal(self.mean(), cov, method="cholesky")

    def update(self, x: np.ndarray, r: float) -> None:
        self.A += np.outer(x, x) / self.noise_var
        self.b += x * r / self.noise_var
        self.n += 1
        self.reward_sum += r

    def to_dict(self) -> dict:
        return {"dim": self.dim, "prior_mean": self.prior_mean, "prior_var": self.prior_var,
                "noise_var": self.noise_var, "A": self.A.tolist(), "b": self.b.tolist(), "n": self.n,
                "reward_sum": self.reward_sum}

    @classmethod
    def from_dict(cls, d: dict) -> "ArmPosterior":
        p = cls(dim=d["dim"], prior_mean=d["prior_mean"], prior_var=d["prior_var"], noise_var=d["noise_var"],
                A=np.array(d["A"]), b=np.array(d["b"]), n=d["n"], reward_sum=d["reward_sum"])
        return p


class LinearTS:
    def __init__(self, dim: int, optimistic_mean: float = 1.0, llm_prior_mean: float = 0.5,
                 prior_var: float = 1.0, noise_var: float = 0.25, seed: int = 0):
        self.dim = dim + 1                 # +1 constant feature
        self.optimistic_mean = optimistic_mean
        self.llm_prior_mean = llm_prior_mean
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.rng = np.random.default_rng(seed)
        self.arms: Dict[str, ArmPosterior] = {}
        self.ensure_arm(CALL_LLM, llm_prior_mean)
        self.frozen = False

    def _x(self, context: np.ndarray) -> np.ndarray:
        return np.concatenate([context, [1.0]])

    def ensure_arm(self, arm: str, prior_mean: Optional[float] = None) -> None:
        if arm not in self.arms:
            self.arms[arm] = ArmPosterior(self.dim, self.optimistic_mean if prior_mean is None else prior_mean,
                                          self.prior_var, self.noise_var)

    def select(self, context: np.ndarray, candidates: List[str]) -> Tuple[str, Dict[str, float]]:
        x = self._x(context)
        scores: Dict[str, float] = {}
        for a in candidates:
            self.ensure_arm(a)
            w = self.arms[a].mean() if self.frozen else self.arms[a].sample(self.rng)
            scores[a] = float(w @ x)
        best = max(scores, key=scores.get)
        return best, scores

    def posterior_mean_reward(self, arm: str, context: np.ndarray) -> float:
        self.ensure_arm(arm)
        return float(self.arms[arm].mean() @ self._x(context))

    def update(self, arm: str, context: np.ndarray, reward: float) -> None:
        if self.frozen:
            return
        self.ensure_arm(arm)
        self.arms[arm].update(self._x(context), reward)

    def to_dict(self) -> dict:
        return {"dim": self.dim - 1, "optimistic_mean": self.optimistic_mean, "llm_prior_mean": self.llm_prior_mean,
                "prior_var": self.prior_var, "noise_var": self.noise_var, "frozen": self.frozen,
                "arms": {k: v.to_dict() for k, v in self.arms.items()}}

    @classmethod
    def from_dict(cls, d: dict, seed: int = 0) -> "LinearTS":
        b = cls(d["dim"], d["optimistic_mean"], d["llm_prior_mean"], d["prior_var"], d["noise_var"], seed)
        b.arms = {k: ArmPosterior.from_dict(v) for k, v in d["arms"].items()}
        b.frozen = d.get("frozen", False)
        return b
