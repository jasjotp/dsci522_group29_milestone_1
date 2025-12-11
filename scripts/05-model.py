#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)
# y_train, y_test are assumed to be binary labels (0/1 or two classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_preprocessor",
        type=str,
        required=True,
        help="Path to the preprocessed dataset pickle (X_train, X_test, y_train, y_test)",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        required=True,
        help="Where to save the trained model pickle file",
    )
    args = parser.parse_args()

    input_path = Path(args.input_preprocessor)
    output_model_path = Path(args.output_model)

    print(f"Loading preprocessed data from: {input_path}")
    X_train, X_test, y_train, y_test = joblib.load(input_path)

    # -------------------------
    # train-logistic-regression
    # -------------------------
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # -------------------------
    # save-trained-model
    # -------------------------
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving trained model to: {output_model_path}")
    joblib.dump(model, output_model_path)

    # -------------------------
    # evaluation / reporting
    # -------------------------

    # predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # probabilit
