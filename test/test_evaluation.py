import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from scripts.functions.evaluation import evaluate_binary_classifier


def test_evaluate_binary_classifier_outputs():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    metrics, fig = evaluate_binary_classifier(
        model, X_train, X_test, y_train, y_test
    )

    assert set(metrics.keys()) == {"train_accuracy", "test_accuracy", "test_roc_auc"}

    assert 0.0 <= metrics["train_accuracy"] <= 1.0
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert 0.0 <= metrics["test_roc_auc"] <= 1.0

    assert hasattr(fig, "savefig")
