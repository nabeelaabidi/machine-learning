import pandas as pd
import numpy as np
import timeit
from clustering import get_cluster_data, generate_validation_plots
from clustering import clustering_experiment, generate_cluster_plots
from clustering import generate_component_plots
from helpers import get_abspath, save_array
from sklearn.decomposition import FastICA
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn


def ica_experiment(X, name, dims):
    """Run ICA on specified dataset and saves mean kurtosis results as CSV
    file.

    Args:
        X (Numpy.Array): Attributes.
        name (str): Dataset name.
        dims (list(int)): List of component number values.

    """
    ica = FastICA(random_state=0, max_iter=5000)
    kurt = {}

    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        df = pd.DataFrame(tmp)
        df = df.kurt(axis=0)
        kurt[dim] = df.abs().mean()

    res = pd.DataFrame.from_dict(kurt, orient='index')
    res.rename(columns={0: 'kurtosis'}, inplace=True)

    # save results as CSV
    resdir = 'results/ICA'
    resfile = get_abspath('{}_kurtosis.csv'.format(name), resdir)
    res.to_csv(resfile, index_label='n')


def save_ica_results(X, name, dims):
    """Run ICA and save projected dataset as CSV.

    Args:
        X (Numpy.Array): Attributes.
        name (str): Dataset name.
        dims (int): Number of components.

    """
    # transform data using ICA
    ica = FastICA(random_state=0, max_iter=5000, n_components=dims)
    res = ica.fit_transform(X)

    # save results file
    resdir = 'results/ICA'
    resfile = get_abspath('{}_projected.csv'.format(name), resdir)
    save_array(array=res, filename=resfile, subdir=resdir)


def generate_kurtosis_plot(name):
    """Plots mean kurtosis as a function of number of components.

    Args:
        name (str): Dataset name.

    """
    resdir = 'results/ICA'
    df = pd.read_csv(get_abspath('{}_kurtosis.csv'.format(name), resdir))

    # get figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))

    # plot explained variance and cumulative explain variance ratios
    x = df['n']
    kurt = df['kurtosis']
    ax.plot(x, kurt, marker='.', color='g')
    ax.set_title('ICA Mean Kurtosis ({})'.format(name))
    ax.set_ylabel('Mean Kurtosis')
    ax.set_xlabel('# Components')
    ax.grid(color='grey', linestyle='dotted')

    # change layout size, font size and width
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)

    # save figure
    plotdir = 'plots/ICA'
    plotpath = get_abspath('{}_kurtosis.png'.format(name), plotdir)
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
    get_cluster_data(aX, aY, 'abalone', km_k=10, gmm_k=5, rdir=rdir)

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
    print 'Running ICA experiments'
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
    rdir = 'results/ICA'
    pdir = 'plots/ICA'

    # generate ICA results
    ica_experiment(dX, 'digits', dims=range(1, wDims + 1))
    ica_experiment(aX, 'abalone', dims=range(1, sDims + 1))

    # generate ICA kurtosis plots
    generate_kurtosis_plot('digits')
    generate_kurtosis_plot('abalone')

    # save ICA results with best # of components
    save_ica_results(dX, 'digits', dims=53)
    save_ica_results(aX, 'abalone', dims=7)

    # re-run clustering experiments
    run_clustering(dY, aY, rdir, pdir)

    # calculate and print running time
    end_time = timeit.default_timer()
    elapsed = end_time - start_time
    print "Completed ICA experiments in {} seconds".format(elapsed)


if __name__ == '__main__':
    main()
