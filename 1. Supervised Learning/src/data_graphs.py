import numpy as np
import pandas as pd
import seaborn as sns
from helpers import get_abspath
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def histogram(labels, dataname, outfile, outpath='plots/datasets'):
    """Generates a histogram of class labels in a given dataset and saves it
    to an output folder in the project directory.

    Args:
        labels (numpy.Array): array containing class labels.
        dataname (str): name of datasets (e.g. abalonequality).
        outfile (str): name of output file name.
        outpath (str): project folder to save plot file.
    """
    # get number of bins
    bins = len(np.unique(labels))

    # set figure params
    sns.set(font_scale=1.3, rc={'figure.figsize': (8, 8)})

    # create plot and set params
    fig, ax = plt.subplots()
    ax.hist(labels, bins=bins)
    fig.suptitle('Class frequency in ' + dataname)
    ax.set_xlabel('Class')
    ax.set_ylabel('Frequency')

    # save plot
    plt.savefig(get_abspath(outfile, outpath))
    plt.close()


def correlation_matrix(df, outfile, outpath='plots/datasets'):
    """ Generates a correlation matrix of all features in a given dataset

    Args:
        df (pandas.DataFrame): Source dataset.
    """
    # format data
    correlations = df.corr()
    names = list(df.columns.values)

    # set figure params
    sns.set(font_scale=0.8, rc={'figure.figsize': (20, 10)})

    # plot correlation heatmap
    sns.heatmap(correlations,
                annot=True,
                linewidth=0,
                xticklabels=names,
                yticklabels=names)
    plt.xticks(rotation=30)

    # save plot
    plt.savefig(get_abspath(outfile, outpath))
    plt.close()


if __name__ == '__main__':
    # load datasets
    p_abalone = get_abspath('abalone.csv', 'data/experiments')
    p_digits = get_abspath('digits.csv', 'data/experiments')
    df_abalone = pd.read_csv(p_abalone)
    df_digits = pd.read_csv(p_digits)

    # generate correlation matrices
    correlation_matrix(df_abalone, 'correlation_abalone.png')
    correlation_matrix(df_digits, 'correlation_digits.png')

    # generate histograms
    histogram(df_abalone['class'], 'abalone', 'hist_abalone.png')
    histogram(df_digits['class'], 'letter', 'hist_digits.png')
