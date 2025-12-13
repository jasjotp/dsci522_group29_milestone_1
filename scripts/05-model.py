from pathlib import Path
import pickle

import click
import joblib
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    RocCurveDisplay,
)


@click.command()
@click.argument("input_preprocessor", type=click.Path(exists=True))
@click.argument("output_model", type=click.Path())
def main(input_preprocessor, output_model):
    """
    Trains a DummyClassifier baseline and a Logistic Regression model.
    Saves trained models and exports roc curve plot as a pickled matplotlib figure.
    """
    input_path = Path(input_preprocessor)
    output_model_path = Path(output_model)

    click.echo(f"Loading preprocessed data from: {input_path}")
    X_train, X_test, y_train, y_test = joblib.load(input_path)

    # train models

    click.echo("Training Dummy Classifier (baseline)...")
    dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy_model.fit(X_train, y_train)

    click.echo("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)


    # save models

    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_output_path = output_model_path.with_name("dummy_model.pkl")
    click.echo(f"Saving dummy model to: {dummy_output_path}")
    joblib.dump(dummy_model, dummy_output_path)

    click.echo(f"Saving trained model to: {output_model_path}")
    joblib.dump(model, output_model_path)


    # evaluation (logreg)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_test_prob)

    click.echo(f"Train accuracy: {train_acc:.3f}")
    click.echo(f"Test accuracy:  {test_acc:.3f}")
    click.echo(f"Test ROC AUC:   {test_roc_auc:.3f}")

    # roc curve plot -> pickle

    plots_dir = output_model_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_test_prob, ax=ax_roc)
    ax_roc.set_title("Logistic Regression – ROC Curve")
    plt.tight_layout()

    roc_pickle_path = plots_dir / "roc_curve.pkl"
    click.echo(f"Saving ROC curve figure to: {roc_pickle_path}")
    with open(roc_pickle_path, "wb") as f:
        pickle.dump(fig_roc, f)

    plt.show()


if __name__ == "__main__":
    main()
