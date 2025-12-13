# test/preprocessing_test.py

import os
import sys
import pandas as pd

# appends the parent directory to the system path so imports work from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.functions.preprocessing import clean_data, split_and_save_data


def test_clean_data(tmp_path):
    df = pd.DataFrame(
        {
            "subject_age": [25.2, 30.9, 19.1, 40.0, 22.5],
            "subject_income_category": ["low", "mid", "high", "low", "mid"],
            "married": ["yes", "no", "no", "yes", "no"],
            "relationship_quality": ["good", "poor", "excellent", "fair", "good"],
        }
    )

    X, y = clean_data(
        df=df,
        features=["subject_age", "subject_income_category", "married"],
        target="relationship_quality",
        int_features=["subject_age"],
        categorical_feature="subject_income_category",
        category_order=["low", "mid", "high"],
    )

    assert pd.api.types.is_integer_dtype(X["subject_age"])
    assert list(X["subject_income_category"].cat.categories) == ["low", "mid", "high"]


def test_split_and_save_data(tmp_path):
    df = pd.DataFrame(
        {
            "subject_age": [20, 21, 22, 23, 24],
            "subject_income_category": ["low", "mid", "high", "low", "mid"],
            "married": ["no", "no", "yes", "no", "yes"],
            "relationship_quality": ["poor", "fair", "good", "excellent", "good"],
        }
    )

    X = df[["subject_age", "subject_income_category", "married"]].copy()
    y = df["relationship_quality"].copy()

    out_dir = tmp_path / "processed"

    split_and_save_data(df=df, X=X, y=y, output_path=out_dir)

    assert (out_dir / "X_train.csv").exists()
    assert (out_dir / "X_test.csv").exists()
    assert (out_dir / "y_train.csv").exists()
    assert (out_dir / "y_test.csv").exists()
