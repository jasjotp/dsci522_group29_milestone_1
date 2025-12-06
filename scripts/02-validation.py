import pandera.pandas as pa
import pointblank as pb
from pathlib import Path
import os 
import pandas as pd 
import click 

@click.command()
@click.argument("input_csv_path")      
@click.argument("output_csv_path")  

def data_validation(input_csv_path, output_csv_path):
    #### Data Validation 1: Correct data file format

    # check file exists
    p = Path(input_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"data file not found at {p.resolve()}")

    # check correct extension
    if p.suffix.lower() != ".csv":
        raise ValueError(f"expected '.csv' extension, got {p.suffix}")

    click.echo('SUCCESS:    Correct data file format test passed')

    # read file
    hcmst = pd.read_csv(p)

    # use pandera to validate that this object is a dataframe
    file_schema = pa.DataFrameSchema({})
    hcmst = file_schema.validate(hcmst)
    click.echo('SUCCESS:    Object read in was successfully converted to a dataframe')

    #### Data Validation 2: Correct column names

    expected_columns = [
        "subject_age",
        "subject_education",
        "subject_sex",
        "subject_ethnicity",
        "subject_income_category",
        "subject_employment_status",
        "same_sex_couple",
        "married",
        "sex_frequency",
        "flirts_with_partner",
        "fights_with_partner",
        "relationship_duration",
        "children",
        "rel_change_during_pandemic",
        "inc_change_during_pandemic",
        "subject_had_covid",
        "partner_had_covid",
        "subject_vaccinated",
        "partner_vaccinated",
        "agree_covid_approach",
        "relationship_quality",
    ]

    actual_columns = list(hcmst.columns)
    if set(actual_columns) != set(expected_columns):
        raise ValueError(
            f"unexpected columns.\nexpected:\n{expected_columns}\n\nfound:\n{actual_columns}"
        )
    column_schema = pa.DataFrameSchema(
        {
            name: pa.Column(nullable=True)  # missingness is checked later
            for name in expected_columns
        },
        strict=True  # fail if there are extra columns
    )

    hcmst = column_schema.validate(hcmst)
    click.echo('SUCCESS:    All columns equal expected columns')

    #### Data Validation 3: No empty observations

    no_empty_rows_schema = pa.DataFrameSchema(
        {},  # column rules checked earlier
        checks=[
            pa.Check(
                lambda df: not (df.isna().all(axis=1)).any(),
                error="found at least one completely empty row in hcmst"
            )
        ]
    )

    hcmst = no_empty_rows_schema.validate(hcmst)
    click.echo('SUCCESS:    No empty rows found in dataframe')

    #### Data Validation 4: Missingness not beyond expected threshold

    # **Note**: The missingness validation in the next cell is applied **only to the columns used in the model** (`subject_age`, `subject_income_category`, `married`, `relationship_duration`, `children`, `relationship_quality`). 

    missing_threshold = 0.05

    cols_to_check_missing = [
        "subject_age",
        "subject_income_category",
        "married",
        "relationship_duration",
        "children",
        "relationship_quality",
    ]

    missingness_schema = pa.DataFrameSchema(
        {
            col: pa.Column(
                nullable=True,
                checks=pa.Check(
                    lambda s: s.isna().mean() <= missing_threshold,
                    element_wise=False,
                    error=(
                        f"column '{col}' has more than "
                        f"{missing_threshold:.0%} missing values"
                    ),
                ),
            )
            for col in cols_to_check_missing
        }
    )

    hcmst = missingness_schema.validate(hcmst)
    click.echo('SUCCESS:    Number of missing values is not beyond expected threshold')

    #### Data Validation 5: See below

    #### Data Validation 6: Check for Duplicate Rows

    # check for duplicate rows using the rows_distinct() function. Got syntax help from: https://posit-dev.github.io/pointblank/reference/Validate.rows_distinct.html#pointblank.Validate.rows_distinct
    pb.Validate(
        data = hcmst
    ).rows_distinct().interrogate()

    click.echo('SUCCESS:    No duplicate rows detected')

    #### Data Validation 7: Check for Outliers

    # **Note**: The below outlier validation in the next cell is applied **only to the columns used in the model** (`subject_age`, `subject_income_category`, `married`, `relationship_duration`, `children`, `relationship_quality`). 

    # '''
    # Perform checks to confirm that outliers do not exist in the data.
    # - subject_age should be between reasonable range, from 0 to 117 (oldest person in world)
    # - subject_income_category should contain values in the existing income_order list above
    # - married should contain values in {married, not_married}
    # - relationship_duration should contain values >= 0 
    # - children should contain values > 0 and < 80
    # - relationship_quality should contain values in: {excellent, good, fair, poor, very_poor}

    # Used the following for syntax help for col_vals_between: https://posit-dev.github.io/pointblank/reference/Validate.col_vals_between.html#pointblank.Validate.col_vals_between
    # Used the following link for syntax for col_vals_in_set: https://posit-dev.github.io/pointblank/reference/Validate.col_vals_in_set.html#pointblank.Validate.col_vals_in_set
    # Used the following link for syntax help to check if relationship_duration >= threshold: https://posit-dev.github.io/pointblank/reference/Validate.col_vals_ge.html
    # '''


    income_order = [
        'under_5k',
        '5k_7k',
        '7k_10k',
        '10k_12k',
        '12k_15k',
        '15k_20k',
        '20k_25k',
        '25k_30k',
        '30k_35k',
        '35k_40k',
        '40k_50k',
        '50k_60k',
        '60k_75k',
        '75k_85k',
        '85k_100k',
        '100k_125k',
        '125k_150k',
        '150k_175k',
        '175k_200k',
        '200k_250k',
        'over_250k'
    ]

    # check for outliers by checking if ranges are in between the above or if str columns are within the allowed set 
    outlier_validation = pb.Validate(data = hcmst) \
        .col_vals_between(
            columns = 'subject_age',
            left = 0,
            right = 117
        ).col_vals_in_set(
            columns = 'subject_income_category',
            set = income_order
        ).col_vals_in_set(
            columns = 'married',
            set = ['married', 'not_married']
        ).col_vals_ge(
            columns = 'relationship_duration',
            value = 0
        ).col_vals_between(
            columns = 'children',
            left = 0,
            right = 80
        ).col_vals_in_set(
            columns = 'relationship_quality',
            set = ['excellent', 'good', 'fair', 'poor', 'very_poor']
    ).interrogate()

    # if all outlier valididation tests pass (are True): log the success
    if outlier_validation.all_passed():
        click.echo('SUCCESS:    Outlier Validation passed')
    else:
        click.echo('FAILED:    At least one Outlier Validation failed')

    #### Data Validation 8: Correct category levels

    object_column_level = (pb.Validate(data = hcmst)
                        .col_vals_in_set(columns = 'subject_income_category',
                                            set = income_order)
                        .col_vals_in_set(columns = 'married', set = ['married', 'not_married'])
                        .col_vals_in_set(columns = 'relationship_quality', set = ['excellent', 'good', 'fair', 'poor', 'very_poor'])
    ).interrogate()
    
    # if all category level validation tests pass (are True): log the success
    if object_column_level.all_passed():
        click.echo('SUCCESS:    Category level tests passed')
    else:
        click.echo('FAILED:    At least one Category level tests failed')

    #### Data Validation 9: Target/response variable follows expected distribution

    response_distribution = pb.Validate(data = hcmst).col_vals_expr(
        expr=lambda df: (
                df["relationship_quality"].value_counts(normalize=True).get("excellent", 0) <= 0.6
                and
                df["relationship_quality"].value_counts(normalize=True).get("good", 0) <= 0.4
                and
                df["relationship_quality"].value_counts(normalize=True).get("fair", 0) <= 0.2
                and
                df["relationship_quality"].value_counts(normalize=True).get("poor", 0) <= 0.2
                and
                df["relationship_quality"].value_counts(normalize=True).get("very_poor", 0) <= 0.2
        )
    ).interrogate()
    
    # if all category level validation tests pass (are True): log the success
    if response_distribution.all_passed():
        click.echo('SUCCESS:    Target/response variable follows expected distribution')
    else:
        click.echo('FAILED:    Target/response variable DOES NOT follow expected distribution')

    # Data validation 10: N.A.

    #### Data Validation 11: No anomalous correlations between features/explanatory variables

    def check_max_correlation(df):
        corr_matrix = df.select_dtypes(include='number').corr().abs()
        pairs = corr_matrix.unstack()
        pairs = pairs[pairs < 1.0]
        return pairs.max() < 0.90

    schema = pa.DataFrameSchema(
        checks=pa.Check(
            check_max_correlation,
            error="High Multicollinearity detected (>0.90)."
        )
    )

    hcmst = schema.validate(hcmst)

    click.echo('SUCCESS:    No High Multicollinearity correlations between features/explanatory variables')

    # write the validated csv to an output file 
    hcmst.to_csv(output_csv_path, index = False)

if __name__ == '__main__':
    data_validation()