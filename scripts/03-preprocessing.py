from pathlib import Path
import click 
import pandas as pd
from joblib import dump
import os
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
import pandera as pa
import sys

# appends the parent directory to the system path so we can import files from parent directories
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.functions.preprocessing import clean_data, validate_data, split_and_save_data, build_preprocessor

# set the types of each feature, grouped by transformation 
NUMERIC_FEATURES = ['subject_age', 'relationship_duration', 'children']
ORDINAL_FEATURES = ['subject_income_category']
CATEGORICAL_FEATURES = ['married']

def fit_and_save_preprocessor(X_train, y_train, output_preprocessor):
    preprocessor = build_preprocessor(NUMERIC_FEATURES, ORDINAL_FEATURES, CATEGORICAL_FEATURES)
    preprocessor.fit(X_train, y_train)

    dump(preprocessor, output_preprocessor)

    print(f"Saved preprocessor to {Path(output_preprocessor).resolve()}")

@click.command()
@click.argument("input_csv", type = str)
@click.argument("output_path", type = str)
@click.argument("output_preprocessor", type = str)

def main(input_csv, output_path, output_preprocessor):
    """
    Preprocesses the data to be used in exploratory data analysis.
    """
    df = pd.read_csv(input_csv)

    # Clean Data
    X, y = clean_data(
        df,
        ['subject_age', 'subject_income_category', 'married', 'relationship_duration', 'children'],
        'relationship_quality',
        ['subject_age', 'children'],
        'subject_income_category',
        ['under_5k', '5k_7k', '7k_10k', '10k_12k', '12k_15k', '15k_20k', '20k_25k', '25k_30k', '30k_35k', '35k_40k', '40k_50k', '50k_60k', '60k_75k', '75k_85k', '85k_100k', '100k_125k', '125k_150k', '150k_175k', '175k_200k', '200k_250k', 'over_250k']
    )

    # validation check 5: check that the input predictor features and the target have the correct datatypes after cleaning 
    schema_X = pa.DataFrameSchema(
        {
            'subject_age': pa.Column(int),
            'subject_income_category': pa.Column(pa.Category),
            'married': pa.Column(str),
            'relationship_duration': pa.Column(float), # allow < 5% nulls naturally
            'children': pa.Column(int)
        }
    )

    # got syntax help for SeriesSchema from: https://pandera.readthedocs.io/en/stable/series_schemas.html
    schema_Y = pa.SeriesSchema(
        str,
        name = 'relationship_quality'
    )

    validated_X, validated_y = validate_data(X, y, schema_X, schema_Y)

    X_train, X_test, y_train, y_test = split_and_save_data(df, validated_X, validated_y, output_path)

    fit_and_save_preprocessor(X_train, y_train, output_preprocessor)

if __name__ == "__main__":
    main()