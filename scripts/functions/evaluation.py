import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, RocCurveDisplay


def evaluate_binary_classifier(model, X_train, X_test, y_train, y_test):
    """
    since training is a repetitive step, i have chosen to abstract this function and apply it onto dummy and logreg models.
    we evaluate a fitted binary classifier on train/test data.

    this computes train and test accuracy, test roc/auc, and returns
    a roc curve matplotlib figure.

    Parameters
    ----------
    model :
        A fitted sklearn-style classifier implementing
        predict() and predict_proba().
    X_train, X_test :
        Feature matrices.
    y_train, y_test :
        Binary target vectors.

    Returns
    -------
    metrics : dict
        Dictionary with keys:
        - train_accuracy
        - test_accuracy
        - test_roc_auc
    fig_roc : matplotlib.figure.Figure
        ROC curve figure for the test set.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_test_prob)

    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_test_prob, ax=ax_roc)
    ax_roc.set_title("Logistic Regression – ROC Curve")
    fig_roc.tight_layout()

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_roc_auc": test_roc_auc,
    }

    return metrics, fig_roc
