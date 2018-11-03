from collections import defaultdict
from itertools import product
import timeit
import pandas as pd
import numpy as np
from clustering import get_cluster_data, generate_validation_plots
from clustering import clustering_experiment, generate_cluster_plots
from clustering import generate_component_plots
from helpers import get_abspath, save_array
from helpers import reconstruction_error, pairwise_dist_corr
from sklearn.random_projection import SparseRandomProjection
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn


def rp_experiment(X, name, dims):
    """Run Randomized Projections on specified dataset and saves reconstruction
    error and pairwise distance correlation results as CSV file.

    Args:
        X (Numpy.Array): Attributes.
        name (str): Dataset name.
        dims (list(int)): List of component number values.

    """
    re = defaultdict(dict)
    pdc = defaultdict(dict)

    for i, dim in product(range(10), dims):
        rp = SparseRandomProjection(random_state=i, n_components=dim)
        rp.fit(X)
        re[dim][i] = reconstruction_error(rp, X)
        pdc[dim][i] = pairwise_dist_corr(rp.transform(X), X)

    re = pd.DataFrame(pd.DataFrame(re).T.mean(axis=1))
    re.rename(columns={0: 'recon_error'}, inplace=True)
    pdc = pd.DataFrame(pd.DataFrame(pdc).T.mean(axis=1))
    pdc.rename(columns={0: 'pairwise_dc'}, inplace=True)
    metrics = pd.concat((re, pdc), axis=1)

    # save results as CSV
    resdir = 'results/RP'
    resfile = get_abspath('{}_metrics.csv'.format(name), resdir)
    metrics.to_csv(resfile, index_label='n')


def save_rp_results(X, name, dims):
    """Run RP and save projected dataset as CSV.

    Args:
        X (Numpy.Array): Attributes.
        name (str): Dataset name.
        dims (int): Number of components.

    """
    # transform data using ICA
    rp = SparseRandomProjection(random_state=0, n_components=dims)
    res = rp.fit_transform(X)

    # save results file
    resdir = 'results/RP'
    resfile = get_abspath('{}_projected.csv'.format(name), resdir)
    save_array(array=res, filename=resfile, subdir=resdir)


def generate_plots(name):
    """Plots reconstruction error and pairwise distance correlation as a
    function of number of components.

    Args:
        name (str): Dataset name.

    """
    resdir = 'results/RP'
    df = pd.read_csv(get_abspath('{}_metrics.csv'.format(name), resdir))

    # get figure and axes
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

    # plot metrics
    x = df['n']
    re = df['recon_error']
    pdc = df['pairwise_dc']
    ax1.plot(x, re, marker='.', color='g')
    ax1.set_title('Reconstruction Error ({})'.format(name))
    ax1.set_ylabel('Reconstruction error')
    ax1.set_xlabel('# Components')
    ax1.grid(color='grey', linestyle='dotted')

    ax2.plot(x, pdc, marker='.', color='b')
    ax2.set_title('Pairwise Distance Correlation ({})'.format(name))
    ax2.set_ylabel('Pairwise distance correlation')
    ax2.set_xlabel('# Components')
    ax2.grid(color='grey', linestyle='dotted')

    # change layout size, font size and width
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)

    # save figure
    plotdir = 'plots/RP'
    plotpath = get_abspath('{}_metrics.png'.format(name), plotdir)
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

    # generate validation plots (relative performance of clustering)
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
    print 'Running RP experiments'
    start_time = timeit.default_timer()

    digitspath = get_abspath('digits.csv', 'data/experiments')
    abalonepath = get_abspath('abalone.csv', 'data/experiments')
    digits = np.loadtxt(digitspath, delimiter=',')
    abalone = np.loadtxt(abalonepath, delimiter=',')

    # split data into X and y
    dX = digits[:, :-1]
    dY = digits[:, -1]
    aX = abalone[:, :-1]
    aY = abalone[:, -1]
    wDims = dX.shape[1]
    sDims = aX.shape[1]
    rdir = 'results/RP'
    pdir = 'plots/RP'

    # generate PCA results
    rp_experiment(dX, 'digits', dims=range(1, wDims + 1))
    rp_experiment(aX, 'abalone', dims=range(1, sDims + 1))

    # generate PCA explained variance plots
    generate_plots(name='digits')
    generate_plots(name='abalone')

    # save ICA results with best # of components
    save_rp_results(dX, 'digits', dims=20)
    save_rp_results(aX, 'abalone', dims=5)

    # re-run clustering experiments
    run_clustering(dY, aY, rdir, pdir)

    # calculate and print running time
    end_time = timeit.default_timer()
    elapsed = end_time - start_time
    print "Completed RP experiments in {} seconds".format(elapsed)


if __name__ == '__main__':
    main()
