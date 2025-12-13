import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer 
from pathlib import Path 

def evaluate_classifier(model, X_train, X_test, y_train, y_test):
    """
    since training is a repetitive step, i have chosen to abstract this function and apply it onto dummy and logreg models.
    we evaluate a fitted classifier on train/test data.

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
        Multi-class target vectors.

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

    y_test_prob = model.predict_proba(X_test)
    test_roc_auc = roc_auc_score(y_test, y_test_prob, multi_class = 'ovr', average = 'micro')

    # use label binarizer to plot micro-average ROC curves
    label_binarizer = LabelBinarizer().fit(y_train)
    y_test_onehot = label_binarizer.transform(y_test)

    fig_roc, ax_roc = plt.subplots()

    RocCurveDisplay.from_predictions(
        y_test_onehot.ravel(), 
        y_test_prob.ravel(),
        name = 'Micro Averaged ROC curve',
        ax = ax_roc
    )
    ax_roc.set_title("Logistic Regression – ROC Curve")
    fig_roc.tight_layout()

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_roc_auc": test_roc_auc,
    }

    return metrics, fig_roc

def save_confusion_matrix(y_true, y_pred, title, output_path):
    '''
    Plots and saves a confusion matrix, from the speciifed observed and predicted y with the specified title to a specific file path.

        Parameters
    ----------
    y_true : array
        True target values.
    y_pred : array
        Predicted target values.
    title : str
        Title of the confusion matrix figure.
    output_path : Path
        File path where the confusion matrix figure will be saved.
    """
    '''

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax = ax
    )

    # set the title and save the figure to the output path 
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
