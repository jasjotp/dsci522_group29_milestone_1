import seaborn as sns
import matplotlib.pyplot as plt

def plot_corr(df, path, title='', annot=True, cmap='coolwarm'):
    corr_mat = df[['subject_age', 'relationship_duration', 'children']].corr()
    sns.heatmap(corr_mat,
                annot=annot,
                cmap=cmap
                ).set_title(title)
    plt.savefig(path)
    plt.close()