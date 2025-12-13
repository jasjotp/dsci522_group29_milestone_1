import seaborn as sns
import matplotlib.pyplot as plt

def make_histogram(df, x, path, bin='auto', title='', xtick_label=None):
    sns.histplot(data=df, x=x, bins=bin
                 ).set_title(title)
    ax = plt.gca()
    ax.set_xticks(range(len(xtick_label)))
    ax.set_xticklabels(xtick_label, rotation=45, ha="right")
    plt.savefig(path)
    plt.close()