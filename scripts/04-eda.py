import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import click
from functions.plot_histogram import plot_histogram
from functions.plot_corr import plot_corr

@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("figure_path", type=click.Path(exists=True))


def main(data_file, figure_path):
    hcmst = pd.read_csv(data_file)

    plot_histogram(hcmst, 'relationship_quality',
                   f'{figure_path}/dist-relationship-quality.png',
                   title='Distribution of Relationship Quality')

    plot_corr(hcmst, f'{figure_path}/corr_plot.png',
              title='Correlation Matrix of Relevant Predictor Variables')

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
    hcmst['subject_income_category'] = pd.Categorical(hcmst['subject_income_category'],
                                                      ordered=True,
                                                      categories=income_order)

    plot_histogram(hcmst, 'subject_income_category',
                   f'{figure_path}/dist-income-category.png',
                   title='Distribution of Income Category',
                   bins=23)


if __name__ == "__main__":
    main()