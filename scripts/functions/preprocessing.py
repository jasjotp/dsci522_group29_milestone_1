from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from pathlib import Path
from sklearn.model_selection import train_test_split

def clean_data(df, features, target, int_features, categorical_feature, category_order):
    '''
    cleans data from a pandas DataFrame by:
        - selecting given feature values for X and a feature value for Y/target
        - casting a feature or list of features to an integer 
        - converting one feature to an ordered categorical variable, with the order specified by the category_order list
    '''
    # select the predictors and the target
    X = df[features].copy()
    y = df[target] # as a series, since the user passes in a str

    # comvert the passed in numeric features to an integer 
    X[int_features] = X[int_features].astype(int)

    # convert the passed in feature to an ordered categorical variable, ordeered by categroy_order
    X[categorical_feature] = X[categorical_feature].astype('category')
    X[categorical_feature] = X[categorical_feature].cat.reorder_categories(
        category_order, ordered = True 
    )

    return X, y


def validate_data(X, y, schema_X, schema_y):
    '''
    validate the datatypes before splitting the data into train and test
    '''
    validated_X = schema_X.validate(X)
    validated_y = schema_y.validate(y)

    return validated_X, validated_y

def split_and_save_data(df, X, y, output_path):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_df_path = out_dir / "hcmst_processed.csv"
    X_train_path = out_dir / "X_train.csv"
    X_test_path = out_dir / "X_test.csv"
    y_train_path = out_dir / "y_train.csv"
    y_test_path = out_dir / "y_test.csv"

    df.to_csv(processed_df_path, index=False)
    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_train.to_frame().to_csv(y_train_path, index=False)
    y_test.to_frame().to_csv(y_test_path, index=False)

    return X_train, X_test, y_train, y_test

def build_preprocessor(numeric_feats, ordinal_feats, categorical_feats):
    
    # create a column transformer that applies transformations to each type of feature
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_feats),
        (OneHotEncoder(drop='if_binary'), categorical_feats),
        (OrdinalEncoder(), ordinal_feats)
    )
    return preprocessor