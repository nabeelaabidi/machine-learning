import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve, validation_curve
import matplotlib.ticker as mtick


def plot_mult_validation_curve(clfs, param_name, param_range, x, y, cv, x_train, y_train, cv_train, title, svm=0,
                               plot_cv=0):
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Error")
    optimized = 1
    for n, clf in enumerate(clfs):
        train_scores, test_scores = validation_curve(clf, x, y, param_name=param_name, param_range=param_range, cv=cv,
                                                     n_jobs=-1)
        train_scores_mean = 1 - np.mean(train_scores, axis=1)
        test_scores_mean = 1 - np.mean(test_scores, axis=1)

        if optimized == 1:
            label_test = "Testing error (Optimized)"
        else:
            label_test = "Testing error (Pruned)"

        train_line = plt.plot(param_range, train_scores_mean, '--')
        colour = train_line[-1].get_color()
        plt.axvline(x=clf.get_params()[param_name], color=colour)
        plt.plot(param_range, test_scores_mean, label=label_test, color=colour)

        # Plot cross-validation if necessary

        if plot_cv == 1:
            not_using, cv_scores = validation_curve(clf, x_train, y_train, param_name=param_name,
                                                    param_range=param_range,
                                                    cv=cv_train, n_jobs=-1)
            cv_scores_mean = 1 - np.mean(cv_scores, axis=1)
            plt.plot(param_range, cv_scores_mean, '-.', linewidth=2, markersize=4, color=colour)

        optimized *= -1
    plt.legend(loc="best")
    plt.savefig(title)
    plt.close()


def plot_validation_curve(estimator, param_name, param_range, x, y, cv, x_train, y_train, cv_train, title, svm=0):
    plt.figure()
    plt.grid()
    plt.axvline(x=estimator.get_params()[param_name])
    train_scores, test_scores = validation_curve(estimator, x, y, param_name=param_name, param_range=param_range, cv=cv,
                                                 n_jobs=1)
    train_scores_mean = 1 - np.mean(train_scores, axis=1)
    test_scores_mean = 1 - np.mean(test_scores, axis=1)

    not_using, cv_scores = validation_curve(estimator, x_train, y_train, param_name=param_name, param_range=param_range,
                                            cv=cv_train,
                                            n_jobs=1)

    cv_scores_mean = 1 - np.mean(cv_scores, axis=1)

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Error")

    if svm == 0:
        plt.plot(param_range, train_scores_mean, label="Training error", color="r")
        plt.plot(param_range, test_scores_mean, label="Testing error", color="g")
        plt.plot(param_range, cv_scores_mean, label="Cross-validation error", color="b")
    else:
        plt.semilogx(param_range, train_scores_mean, label="Training error", color="r")
        plt.semilogx(param_range, test_scores_mean, label="Testing error", color="g")
        plt.semilogx(param_range, cv_scores_mean, label="Cross-validation error", color="b")

    plt.legend(loc="best")
    plt.savefig(title)
    plt.close()


def plot_learning_curve(estimator, title, X, y, dn, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")  
    plotdir = 'plots'
    plot_tgt = '{}/{}'.format(plotdir, title)
    plotpath = get_abspath('{}_LC.png'.format(dn), plot_tgt)
    plt.savefig(plotpath)
    plt.close()



def feature_importance_plot(estimator, features_names, title):
    feature_importance = estimator.feature_importances_
    # make importance relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, features_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title(title)
    plt.savefig(title)
    plt.close()
