import pandas as pd
import numpy as np
import timeit
from clustering import get_cluster_data, generate_validation_plots
from clustering import clustering_experiment, generate_cluster_plots
from clustering import generate_component_plots
from helpers import get_abspath, save_array
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn


def rf_experiment(X, y, name, theta):
    """Run RF on specified dataset and saves feature importance metrics and best
    results CSV.

    Args:
        X (Numpy.Array): Attributes.
        y (Numpy.Array): Labels.
        name (str): Dataset name.
        theta (float): Min cumulative information gain threshold.

    """
    rfc = RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=0)
    fi = rfc.fit(X, y).feature_importances_

    # get feature importance and sort by value in descending order
    i = [i + 1 for i in range(len(fi))]
    fi = pd.DataFrame({'importance': fi, 'feature': i})
    fi.sort_values('importance', ascending=False, inplace=True)
    fi['i'] = i
    cumfi = fi['importance'].cumsum()
    fi['cumulative'] = cumfi

    # generate dataset that meets cumulative feature importance threshold
    idxs = fi.loc[:cumfi.where(cumfi > theta).idxmin(), :]
    idxs = list(idxs.index)
    reduced = X[:, idxs]

    # save results as CSV
    resdir = 'results/RF'
    fifile = get_abspath('{}_fi.csv'.format(name), resdir)
    resfile = get_abspath('{}_projected.csv'.format(name), resdir)
    save_array(array=reduced, filename=resfile, subdir=resdir)
    fi.to_csv(fifile, index_label=None)


def generate_fi_plot(name, theta):
    """Plots feature importance and cumulative feature importance values sorted
    by feature index.

    Args:
        name (str): Dataset name.
        theta (float): Explained variance percentage threshold.

    """
    resdir = 'results/RF'
    df = pd.read_csv(get_abspath('{}_fi.csv'.format(name), resdir))

    # get figure and axes
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))

    # plot explained variance and cumulative explain variance ratios
    ax2 = ax1.twinx()
    x = df['i']
    fi = df['importance']
    cumfi = df['cumulative']
    ax1.bar(x, height=fi, color='b', tick_label=df['feature'], align='center')
    ax2.plot(x, cumfi, color='r', label='Cumulative Info Gain')
    ax1.set_title('Feature Importance ({})'.format(name))
    ax1.set_ylabel('Gini Gain')
    ax2.set_ylabel('Cumulative Gini Gain')
    ax1.set_xlabel('Feature Index')
    ax2.axhline(y=theta, linestyle='--', color='r')
    ax1.grid(b=None)
    ax2.grid(b=None)

    # change layout size, font size and width
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)

    # save figure
    plotdir = 'plots/RF'
    plotpath = get_abspath('{}_fi.png'.format(name), plotdir)
    plt.savefig(plotpath)
    plt.clf()


def run_clustering(dY, aY, rdir, pdir):
    """Re-run clustering experiments on datasets after dimensionality
    reduction.

    Args:
        dY (Numpy.Array): Labels for digits.
        aY (Numpy.Array): Labels for abalone.
        rdir (str): Input file directory.
        pdir (str): Output directory.

    """
    digitspath = get_abspath('digits_projected.csv', rdir)
    abalonepath = get_abspath('abalone_projected.csv', rdir)
    dX = np.loadtxt(digitspath, delimiter=',')
    aX = np.loadtxt(abalonepath, delimiter=',')
    rdir = rdir + '/clustering'
    pdir = pdir + '/clustering'

    # re-run clustering experiments after applying PCA
    clusters = [2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 50]
    clustering_experiment(dX, dY, 'digits', clusters, rdir=rdir)
    clustering_experiment(aX, aY, 'abalone', clusters, rdir=rdir)

    # generate 2D data for cluster visualization
    get_cluster_data(dX, dY, 'digits', km_k=10, gmm_k=10, rdir=rdir)
    get_cluster_data(aX, aY, 'abalone', km_k=5, gmm_k=10, rdir=rdir)

    # generate component plots (metrics to choose size of k)
    generate_component_plots(name='digits', rdir=rdir, pdir=pdir)
    generate_component_plots(name='abalone', rdir=rdir, pdir=pdir)

    # # generate validation plots (relative performance of clustering)
    generate_validation_plots(name='digits', rdir=rdir, pdir=pdir)
    generate_validation_plots(name='abalone', rdir=rdir, pdir=pdir)

    # generate validation plots (relative performance of clustering)
    df_digits = pd.read_csv(get_abspath('digits_2D.csv', rdir))
    df_abalone = pd.read_csv(get_abspath('abalone_2D.csv', rdir))
    generate_cluster_plots(df_digits, name='digits', pdir=pdir)
    generate_cluster_plots(df_abalone, name='abalone', pdir=pdir)


def main():
    """Run code to generate results.

    """
    print 'Running RF experiments'
    start_time = timeit.default_timer()

    digitspath = get_abspath('digits.csv', 'data/experiments')
    abalonepath = get_abspath('abalone.csv', 'data/experiments')
    digits = np.loadtxt(digitspath, delimiter=',')
    abalone = np.loadtxt(abalonepath, delimiter=',')

    # set cumulative feature importance threshold
    theta = 0.80

    # split data into X and y
    dX = digits[:, :-1]
    dY = digits[:, -1]
    aX = abalone[:, :-1]
    aY = abalone[:, -1]
    wDims = dX.shape[1]
    sDims = aX.shape[1]
    rdir = 'results/RF'
    pdir = 'plots/RF'

    # generate RF results
    rf_experiment(dX, dY, 'digits', theta=theta)
    rf_experiment(aX, aY, 'abalone', theta=theta)

    # generate RF feature importance plots
    generate_fi_plot('digits', theta=theta)
    generate_fi_plot('abalone', theta=theta)

    # re-run clustering experiments
    run_clustering(dY, aY, rdir, pdir)

    # calculate and print running time
    end_time = timeit.default_timer()
    elapsed = end_time - start_time
    print "Completed RF experiments in {} seconds".format(elapsed)


if __name__ == '__main__':
    main()
