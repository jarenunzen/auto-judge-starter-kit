import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from typing import List, Tuple
import re


# --- 1. Feature extraction (mirrors your judge logic) ---

def extract_features(score1: int, score2: int) -> np.ndarray:
    """
    Build a feature vector from the two LLM scores.
    No gold labels used here — purely structural signal.
    """
    difference = abs(score1 - score2)
    return np.array([
        score1,                        # raw positive relevance
        score2,                        # raw negation relevance
        difference,                    # agreement gap
        score1 - score2,               # signed gap (direction matters)
        score1 / (score2 + 1),         # ratio (avoid div/0)
        int(difference == 0),          # flag: no separation
        int(difference == 1),          # flag: weak separation
        int(difference == 2),          # flag: moderate separation
        int(difference == 3),          # flag: strong separation
    ], dtype=float)


# --- 2. Penalty optimizer (no gold data) ---

class PenaltyOptimizer:
    """
    Learns optimal per-bucket penalties using a held-out dev set of
    (score1, score2, human_relevance) triples.

    'human_relevance' is your dev-set ground truth (e.g. TREC qrels),
    used ONLY for optimization — never leaked into the judge at inference time.
    """

    def __init__(self):
        self.penalties_ = {0: 0.0, 1: 0.1, 2: 0.02, 3: 0.0}  # difference -> penalty

    def _apply_penalties(self, scores_pairs: List[Tuple[int, int]], penalties: np.ndarray) -> np.ndarray:
        """
        Apply a candidate set of 4 penalties (one per difference bucket)
        and return predicted rel_scores.
        """
        results = []
        for s1, s2 in scores_pairs:
            diff = abs(s1 - s2)
            penalty = penalties[diff]  # index 0..3
            results.append(s1 - penalty)
        return np.array(results)

    def _loss(self, penalties: np.ndarray, scores_pairs: List[Tuple[int, int]], targets: np.ndarray) -> float:
        """
        MSE against dev-set relevance labels.
        Add a monotonicity constraint: penalty should decrease as difference grows.
        """
        predicted = self._apply_penalties(scores_pairs, penalties)
        mse = np.mean((predicted - targets) ** 2)

        # Soft monotonicity: penalize if lower difference has smaller penalty
        # i.e. we expect penalties[0] >= penalties[1] >= penalties[2] >= penalties[3]
        mono_violations = (
            max(0, penalties[1] - penalties[0]) +  # diff=1 should cost more than diff=3
            max(0, penalties[2] - penalties[1]) +
            max(0, penalties[3] - penalties[2])
        )

        return mse + 0.5 * mono_violations

    def fit(self, scores_pairs: List[Tuple[int, int]], targets: np.ndarray):
        """
        scores_pairs: list of (score1, score2) from your LLM judge
        targets: gold relevance grades from qrels (dev set only)
        """
        x0 = np.array([0.15, 0.1, 0.02, 0.0])  # initial guess: [diff=0, diff=1, diff=2, diff=3]

        bounds = [(0.0, 0.5)] * 4  # penalties must be non-negative and reasonable

        result = minimize(
            self._loss,
            x0,
            args=(scores_pairs, targets),
            method="L-BFGS-B",
            bounds=bounds,
        )

        opt_penalties = result.x
        self.penalties_ = {i: opt_penalties[i] for i in range(4)}
        print(f"Optimized penalties by difference bucket:")
        for diff, pen in self.penalties_.items():
            print(f"  diff={diff}: penalty={pen:.4f}")
        return self

    def predict(self, score1: int, score2: int) -> float:
        diff = abs(score1 - score2)
        return score1 - self.penalties_[diff]


# --- 3. Optional: richer model using Ridge regression on features ---

class FeatureBasedRelevanceModel:
    """
    If you want the model to learn non-linear interactions beyond
    just the difference bucket, use polynomial Ridge regression.
    """

    def __init__(self, degree: int = 2):
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("ridge", Ridge(alpha=1.0)),
        ])

    def fit(self, scores_pairs: List[Tuple[int, int]], targets: np.ndarray):
        X = np.array([extract_features(s1, s2) for s1, s2 in scores_pairs])
        self.model.fit(X, targets)
        return self

    def predict(self, score1: int, score2: int) -> float:
        x = extract_features(score1, score2).reshape(1, -1)
        return float(self.model.predict(x)[0])


# --- 4. Drop-in replacement for your judge's scoring logic ---

def make_rel_score(score1: int, score2: int, model: PenaltyOptimizer) -> float:
    return model.predict(score1, score2)