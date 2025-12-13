from pathlib import Path
import pickle

import click
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from functions.evaluation import evaluate_classifier, save_confusion_matrix


@click.command()
@click.argument("processed_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("input_preprocessor", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_model", type=click.Path())
def main(processed_dir, input_preprocessor, output_model):
    processed_dir = Path(processed_dir)
    preprocessor_path = Path(input_preprocessor)
    output_model_path = Path(output_model)

    # load split data
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze("columns")

    # load and apply preprocessor
    preprocessor = joblib.load(preprocessor_path)

    # do not fit on train again, the saved preprocessor was already fitted in step 3 transform train and test
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # train models
    click.echo("Training Dummy Classifier (baseline)...")
    dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy_model.fit(X_train_t, y_train)

    click.echo("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_t, y_train)

    # save models
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_output_path = output_model_path.with_name("dummy_model.pkl")
    joblib.dump(dummy_model, dummy_output_path)
    joblib.dump(model, output_model_path)

    # evaluation (logreg) + roc curve fig
    metrics, fig_roc = evaluate_classifier(
        model, X_train_t, X_test_t, y_train, y_test
    )

    click.echo(f"Train accuracy: {metrics['train_accuracy']:.3f}")
    click.echo(f"Test accuracy:  {metrics['test_accuracy']:.3f}")
    click.echo(f"Test ROC AUC:   {metrics['test_roc_auc']:.3f}")
    
    # save the confusion matrices of the Logisitic Regression model for the training and test set
    figures_dir = Path('figures')
    figures_dir.mkdir(parents = True, exist_ok = True)

    save_confusion_matrix(
        y_true = y_train,
        y_pred = model.predict(X_train_t),
        title = 'Confusion Matrix (Train)',
        output_path = figures_dir / 'model_confusion_train.png'
    )

    save_confusion_matrix(
        y_true = y_test,
        y_pred = model.predict(X_test_t),
        title = 'Confusion Matrix (Test)',
        output_path = figures_dir / 'model_confusion_test.png'
    )
    
    # roc curve plot -> pickle
    plots_dir = output_model_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    roc_pickle_path = plots_dir / "roc_curve.pkl"
    with open(roc_pickle_path, "wb") as f:
        pickle.dump(fig_roc, f)
    
    # save the roc plot locally to the figures directory
    fig_roc.savefig(figures_dir / 'model_roc_micro_ovr.png', bbox_inches = 'tight')
    plt.close(fig_roc)

if __name__ == "__main__":
    main()
