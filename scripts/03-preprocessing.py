from pathlib import Path
import click 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from joblib import dump

@click.command()
@click.argument("input_csv")
@click.argument("output_path")
@click.argument("output_preprocessor")
@click.option('--input_csv', type=str, help="Path of raw csv")
@click.option('--output_path', type=str, help="Path of processed train/test split")
@click.option('--output_preprocessor', type=str, help="Path of fitted preprocessor")
def main(input_csv, output_path, output_preprocessor):
    """
    Preprocesses the data to be used in exploratory data analysis.
    """
    df = pd.read_csv(input_csv)

    # Clean Data

    X = df[['subject_age', 'subject_income_category', 'married', 'relationship_duration', 'children']].copy()
    y = df['relationship_quality']

    # simple data cleaning - change subject age and children from floats to intgers 
    X['subject_age'] = X['subject_age'].astype(int)
    X['children'] = X['children'].astype(int)

    # reorder income category 
    X['subject_income_category'] = X['subject_income_category'].astype('category')
    X['subject_income_category'] = X['subject_income_category'].cat.reorder_categories(
        ['under_5k', '5k_7k', '7k_10k', '10k_12k', '12k_15k', '15k_20k', '20k_25k', '25k_30k', '30k_35k', '35k_40k', '40k_50k', '50k_60k', '60k_75k', '75k_85k', '85k_100k', '100k_125k', '125k_150k', '150k_175k', '175k_200k', '200k_250k', 'over_250k'], ordered = True 
    )

    # split data into train and test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 123
    )

    # after train_test_split
    X_train_path = Path(output_path).parent / "X_train.csv"
    X_test_path = Path(output_path).parent / "X_test.csv"
    y_train_path = Path(output_path).parent / "y_train.csv"
    y_test_path = Path(output_path).parent / "y_test.csv"

    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)

    # y is a Series, so wrap in DataFrame for consistent columns
    y_train.to_frame().to_csv(y_train_path, index=False)
    y_test.to_frame().to_csv(y_test_path, index=False)


    # preprocess the data by applying the appropriate transformations to each
    numeric_features = ['subject_age', 'relationship_duration', 'children']
    ordinal_features = ['subject_income_category']
    categorical_features = ['married']

    # create a column transformer that applies transformations to each type of feature
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(drop='if_binary'), categorical_features),
        (OrdinalEncoder(), ordinal_features)
    )

    preprocessor.fit(X_train)



    # save the cleaned csv
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # save the fitted preprocessor
    dump(preprocessor, output_preprocessor)

    click.echo(f"Saved data to {out_path.resolve()}")
    click.echo(f"Saved preprocessor to {Path(output_preprocessor).resolve()}")

if __name__ == "__main__":
    main()