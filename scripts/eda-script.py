import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

hcmst = pd.read_csv('data/processed/hcmst_clean.csv')

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

sns.histplot(
    data = hcmst,
    x = 'relationship_quality',
    stat = 'count'
).set_title('Distribution of Relationship Quality')
plt.savefig('figures/dist-relationship-quality.png')
plt.close()

corr_mat = hcmst[['subject_age','relationship_duration', 'children']].corr()
sns.heatmap(corr_mat,
            annot=True,
            cmap='coolwarm'
            ).set_title('Correlation Matrix of Relevant Predictor Variables')
plt.savefig('figures/corr_plot.png')
plt.close()

hcmst['subject_income_category'] = pd.Categorical(hcmst['subject_income_category'],
                                                  ordered = True,
                                                  categories = income_order)
sns.histplot(
    data = hcmst,
    x = 'subject_income_category',
    bins = 23
).set_title('Distribution of Income Category')
plt.xticks(rotation = 45, ha = 'right')
plt.tight_layout()
plt.savefig('figures/dist-income-category.png')
plt.close()