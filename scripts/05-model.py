# 05-model.py

import click
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("figure_path", type=click.Path(exists=True))

def main(data_file, figure_path):
    """
    Train & evaluate a logistic regression model.

    INPUT_PATH: path to preprocessed data produced by 03-preprocessing.py
   

    OUTPUT_PREFIX: path/filename prefix for output artifacts
 
    """
    input_path = Path(data_file)
    output_prefix = Path(data_file)

    # Ensure the output directory exists
    output_prefix.parent.mkdir(parents=True, exist_ok=True)


    # 1. Load preprocessed data from 03-preprocessing.py
 
    # Expecting a joblib file with:
    #   data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    data = joblib.load(input_path)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]


    # 2. Fit logistic regression model

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)


    # 3. Training confusion matrix

    fig_train, ax_train = plt.subplots()
    disp_train = ConfusionMatrixDisplay.from_estimator(
        logreg, X_train, y_train, ax=ax_train
    )
    ax_train.grid(False)
    ax_train.set_title(
        "Confusion Matrix of Logistic Regression using training data"
    )

    train_cm_path = output_prefix.with_name(
        output_prefix.name + "_confusion_train.png"
    )
    fig_train.savefig(train_cm_path, bbox_inches="tight")
    plt.close(fig_train)


    # 4. Testing confusion matrix

    fig_test, ax_test = plt.subplots()
    disp_test = ConfusionMatrixDisplay.from_estimator(
        logreg, X_test, y_test, ax=ax_test
    )
    ax_test.grid(False)
    ax_test.set_title(
        "Confusion Matrix of Logistic Regression using testing data"
    )

    test_cm_path = output_prefix.with_name(
        output_prefix.name + "_confusion_test.png"
    )
    fig_test.savefig(test_cm_path, bbox_inches="tight")
    plt.close(fig_test)

    # 5. Micro-average AUC ROC (score)

    y_score = logreg.predict_proba(X_test)
    micro_roc_auc_ovr = roc_auc_score(
        y_test,
        y_score,
        multi_class="ovr",
        average="micro",
    )

    # Print to console as well
    print(
        f"Micro-averaged One-vs-Rest ROC AUC score:\n"
        f"{micro_roc_auc_ovr:.2f}"
    )

    # 6. Micro-averaged AUC ROC plot

    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)

    fig_roc, ax_roc = plt.subplots()
    display = RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        y_score.ravel(),
        name="micro-average OvR",
        color="darkorange", 
        plot_chance_level=True,
        ax=ax_roc,
    )
    ax_roc.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=(
            "Micro-averaged One-vs-Rest\n"
            "Receiver Operating Characteristic"
        ),
    )

    roc_path = output_prefix.with_name(
        output_prefix.name + "_roc_micro_ovr.png"
    )
    fig_roc.savefig(roc_path, bbox_inches="tight")
    plt.close(fig_roc)

    # 7. Save metrics table

    metrics_df = pd.DataFrame(
        {
            "metric": ["micro_roc_auc_ovr"],
            "value": [micro_roc_auc_ovr],
        }
    )

    metrics_path = output_prefix.with_name(
        output_prefix.name + "_metrics.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Saved training confusion matrix to: {train_cm_path}")
    print(f"Saved testing confusion matrix to: {test_cm_path}")
    print(f"Saved ROC curve to: {roc_path}")
    print(f"Saved metrics table to: {metrics_path}")


if __name__ == "__main__":
    main()
