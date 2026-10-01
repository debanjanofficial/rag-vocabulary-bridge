"""
Train and Evaluate the Learned Gating Classifier for Adaptive Query Routing.

Trains a multi-class multinomial Logistic Regression gating network on the
6-dimensional feature representations:
  Phi(q) = [S_term, C_prod, M_prod, J_agree, H_dense, R_drift]

Produces calibrated action posterior probabilities P(action | Phi(q))
across all 5 discrete execution actions in the routing taxonomy.
Saves the learned weights and intercepts to:
  src/router/learned_weights.json
"""

import os
import sys
import json
import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import DATA_DIR
from src.router.features import FeatureVector, extract_features
from src.router.policy import RoutingAction

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT, "src", "router", "learned_weights.json"
)

ACTION_NAMES = [
    RoutingAction.PASSTHROUGH.value,
    RoutingAction.EXPAND_TERMINOLOGY.value,
    RoutingAction.FILTER_METADATA.value,
    RoutingAction.EXPAND_SEMANTIC.value,
    RoutingAction.CLARIFY.value,
]

ACTION_TO_IDX = {name: idx for idx, name in enumerate(ACTION_NAMES)}
IDX_TO_ACTION = {idx: name for idx, name in enumerate(ACTION_NAMES)}


def build_synthetic_boundary_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Generate representative calibrated feature distributions.

    Synthesizes boundary cases across each of the 5 routing actions
    based on calibrated domain priors:
      - PASSTHROUGH: high J_agree, low S_term, low H_dense
      - EXPAND_TERMINOLOGY: high S_term, low R_drift
      - FILTER_METADATA: high C_prod, high M_prod
      - EXPAND_SEMANTIC: low J_agree, low S_term, moderate entropy
      - CLARIFY: high H_dense, low M_prod, moderate C_prod
    """
    np.random.seed(42)
    n_per_class = 200
    features_list: list[np.ndarray] = []
    labels_list: list[int] = []

    # 1. PASSTHROUGH (Idx 0)
    for _ in range(n_per_class):
        s_term = np.random.uniform(0.0, 0.35)
        c_prod = np.random.uniform(0.1, 0.6)
        m_prod = np.random.uniform(0.05, 0.4)
        j_agree = np.random.uniform(0.35, 0.85)
        h_dense = np.random.uniform(0.2, 0.65)
        r_drift = np.random.uniform(0.0, 0.25)
        features_list.append(
            [s_term, c_prod, m_prod, j_agree, h_dense, r_drift]
        )
        labels_list.append(0)

    # 2. EXPAND_TERMINOLOGY (Idx 1)
    for _ in range(n_per_class):
        s_term = np.random.uniform(0.52, 1.0)
        c_prod = np.random.uniform(0.1, 0.6)
        m_prod = np.random.uniform(0.0, 0.35)
        j_agree = np.random.uniform(0.0, 0.28)
        h_dense = np.random.uniform(0.4, 0.98)
        r_drift = np.random.uniform(0.0, 0.45)
        features_list.append(
            [s_term, c_prod, m_prod, j_agree, h_dense, r_drift]
        )
        labels_list.append(1)

    # 3. FILTER_METADATA (Idx 2)
    for _ in range(n_per_class):
        s_term = np.random.uniform(0.0, 0.45)
        c_prod = np.random.uniform(0.68, 0.99)
        m_prod = np.random.uniform(0.25, 0.95)
        j_agree = np.random.uniform(0.1, 0.45)
        h_dense = np.random.uniform(0.2, 0.7)
        r_drift = np.random.uniform(0.0, 0.3)
        features_list.append(
            [s_term, c_prod, m_prod, j_agree, h_dense, r_drift]
        )
        labels_list.append(2)

    # 4. EXPAND_SEMANTIC (Idx 3)
    for _ in range(n_per_class):
        s_term = np.random.uniform(0.0, 0.38)
        c_prod = np.random.uniform(0.05, 0.45)
        m_prod = np.random.uniform(0.0, 0.25)
        j_agree = np.random.uniform(0.0, 0.14)
        h_dense = np.random.uniform(0.3, 0.75)
        r_drift = np.random.uniform(0.0, 0.35)
        features_list.append(
            [s_term, c_prod, m_prod, j_agree, h_dense, r_drift]
        )
        labels_list.append(3)

    # 5. CLARIFY (Idx 4)
    for _ in range(n_per_class):
        s_term = np.random.uniform(0.1, 0.45)
        c_prod = np.random.uniform(0.38, 0.75)
        m_prod = np.random.uniform(0.0, 0.08)
        j_agree = np.random.uniform(0.0, 0.22)
        h_dense = np.random.uniform(0.68, 0.99)
        r_drift = np.random.uniform(0.0, 0.3)
        features_list.append(
            [s_term, c_prod, m_prod, j_agree, h_dense, r_drift]
        )
        labels_list.append(4)

    return (
        np.array(features_list, dtype=float),
        np.array(labels_list, dtype=int),
    )


def train_and_evaluate() -> dict[str, Any]:
    """Train multinomial logistic regression and evaluate with 5-fold CV."""
    x_data, y_data = build_synthetic_boundary_dataset()
    logger.info("Dataset shape: %s samples, 6 features", x_data.shape[0])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores: list[float] = []
    f1_scores: list[float] = []
    loss_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_data, y_data), 1):
        x_tr, y_tr = x_data[train_idx], y_data[train_idx]
        x_va, y_va = x_data[val_idx], y_data[val_idx]

        clf = LogisticRegression(
            C=1.5,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        clf.fit(x_tr, y_tr)
        preds = clf.predict(x_va)
        probs = clf.predict_proba(x_va)

        acc = accuracy_score(y_va, preds)
        f1 = f1_score(y_va, preds, average="macro")
        loss = log_loss(y_va, probs)

        acc_scores.append(acc)
        f1_scores.append(f1)
        loss_scores.append(loss)
        logger.info("Fold %d - Acc: %.4f, Macro-F1: %.4f, Loss: %.4f",
                    fold, acc, f1, loss)

    mean_acc = float(np.mean(acc_scores))
    mean_f1 = float(np.mean(f1_scores))
    mean_loss = float(np.mean(loss_scores))
    logger.info("5-Fold CV Mean Accuracy: %.4f (+/- %.4f)",
                mean_acc, float(np.std(acc_scores)))
    logger.info("5-Fold CV Mean Macro F1: %.4f (+/- %.4f)",
                mean_f1, float(np.std(f1_scores)))
    logger.info("5-Fold CV Mean Log-Loss: %.4f", mean_loss)

    # Fit final model on complete dataset
    final_clf = LogisticRegression(
        C=1.5,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    final_clf.fit(x_data, y_data)

    feature_names = [
        "s_term", "c_prod", "m_prod", "j_agree", "h_dense", "r_drift"
    ]
    model_payload = {
        "model_type": "MultinomialLogisticRegression",
        "action_names": ACTION_NAMES,
        "feature_names": feature_names,
        "weights": final_clf.coef_.tolist(),
        "intercept": final_clf.intercept_.tolist(),
        "metrics": {
            "cv_accuracy": round(mean_acc, 4),
            "cv_macro_f1": round(mean_f1, 4),
            "cv_log_loss": round(mean_loss, 4),
            "n_samples": int(x_data.shape[0]),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(WEIGHTS_PATH)), exist_ok=True)
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=2)

    logger.info("Successfully exported learned weights to: %s", WEIGHTS_PATH)
    return model_payload


def print_interpretable_coefficients(payload: dict[str, Any]) -> None:
    """Print feature coefficient weights for academic interpretability."""
    weights = np.array(payload["weights"])
    intercept = np.array(payload["intercept"])
    features = payload["feature_names"]
    actions = payload["action_names"]

    print("\n" + "=" * 70)
    print("  LEARNED GATING POLICY COEFFICIENTS W in R^(5x6)")
    print("=" * 70)
    feat_hdr = " | ".join(f"{f:>8}" for f in features)
    header = f"{'Action':<22} | {feat_hdr} | Intercept"
    print(header)
    print("-" * len(header))
    for i, act in enumerate(actions):
        c_str = " | ".join(
            f"{weights[i, j]:+8.3f}" for j in range(len(features))
        )
        row_str = f"{act:<22} | {c_str} | {intercept[i]:+8.3f}"
        print(row_str)
    print("=" * 70 + "\n")


def main():
    """Main training execution entrypoint."""
    payload = train_and_evaluate()
    print_interpretable_coefficients(payload)


if __name__ == "__main__":
    main()
