import seaborn as sns
import matplotlib.pyplot as plt

def make_histogram(df, x, path, bin='auto', title=''):
    sns.histplot(data=df, x=x, bins=bin
                 ).set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(path)
    plt.close()