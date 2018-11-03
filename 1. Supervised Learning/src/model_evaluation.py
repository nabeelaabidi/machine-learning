"""
Contains all the necessary functions to evaluate trained models and generate
validation, learning, iteration and timing curves.

"""
from helpers import load_pickled_model, get_abspath
from model_train import split_data, balanced_accuracy, balanced_f1
from sklearn.metrics import make_scorer
from sklearn import metrics
from sklearn.model_selection import learning_curve,  cross_val_score, validation_curve
import pandas as pd
import numpy as np
import timeit
#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import seaborn as sns


def basic_results(grid, X_test, y_test, X_train, y_train, data_name, clf_name):
    """Gets best fit against test data for best estimator from a particular
    grid object. Note: test score funtion is the same scoring function used
    for training.

    Args:
        grid (GridSearchCV object): Trained grid search object.
        X_test (numpy.Array): Test features.
        y_test (numpy.Array): Test labels.
        data_name (str): Name of data set being tested.
        clf_name (str): Type of algorithm.

    """
    # get best score, test score, scoring function and best parameters
    clf = clf_name
    dn = data_name
    bs = grid.best_score_
    ts = grid.score(X_test, y_test)
    sf = grid.scorer_
    bp = grid.best_params_

    be = grid.best_estimator_
    print ('\nClassifier: {}'.format(clf))
    print ('Data Name: {}'.format(dn))
#    print ('Best Parameters: {}'.format(bp))
    
    be.fit(X_train, y_train)
    pred = be.predict(X_test)
    scores = cross_val_score(be, X_train, y_train, cv=5)
#    print(("cross-validation mean: {:.3f} (std: {:.3f})".format(scores.mean(), scores.std())))
#    print((metrics.classification_report(y_test, pred)))
#    print((metrics.confusion_matrix(y_test, pred)))
    

    print("Accuracy Score:{}".format(metrics.accuracy_score(y_true=y_test, y_pred=pred)))
#    print(("Error - {0}".format(1 - metrics.accuracy_score(y_true=y_test, y_pred=pred))))
#    print("Classification Report-")
#    print((metrics.classification_report(y_test, pred)))
#    print("Confusion Matrix")
#    print((metrics.confusion_matrix(y_test, pred)))
#    
    # write results to a combined results file
    parentdir = 'results'
    resfile = get_abspath('combined_results.csv', parentdir)
    with open(resfile, 'a') as f:
        f.write('{}|{}|{}|{}|{}|{}\n'.format(clf, dn, bs, ts, sf, bp))


def plot_learning_curve(estimator, title, X, y, scorer, dn,  cv=6,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
  
    plt.figure()
    plt.title("Neural Network with BackPropogation".format(dn))
    
    
    plt.xlabel("Training Dataset (%)")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
   
    test_scores_mean = np.mean(test_scores, axis=1)
   
    train_size = np.linspace(.1, 1.0, 5) *100
    plt.grid()

   
    plt.plot(train_size,  train_scores_mean, 'o-', color="r",
             label="Training score", lw=2)
    plt.plot(train_size, test_scores_mean, 'o-', color="g",
             label="Cross-validation score", lw=2)

    plt.legend(loc="best")
    
    plotdir = 'plots'
    plot_tgt = '{}/{}'.format(plotdir, title)
    plotpath = get_abspath('{}_LC.png'.format(dn), plot_tgt)
    plt.savefig(plotpath)
    plt.close()
    

def create_timing_curve(estimator, dataset, data_name, clf_name):
    # set training sizes and intervals
    train_sizes = np.arange(0.01, 1.0, 0.03)

    # initialise variables
    train_time = []
    predict_time = []
    df_final = []

    # iterate through training sizes and capture training and predict times
    for i, train_data in enumerate(train_sizes):
        X_train, X_test, y_train, y_test = split_data(
            dataset, test_size=1 - train_data)
        start_train = timeit.default_timer()
        estimator.fit(X_train, y_train)
        end_train = timeit.default_timer()
        estimator.predict(X_test)
        end_predict = timeit.default_timer()
        train_time.append(end_train - start_train)
        predict_time.append(end_predict - end_train)
        df_final.append([train_data, train_time[i], predict_time[i]])

    # save timing results to CSV
    timedata = pd.DataFrame(data=df_final, columns=[
                            'Training Data Percentage', 'Train Time', 'Test Time'])
    resdir = 'results'
    res_tgt = '{}/{}'.format(resdir, clf_name)
    timefile = get_abspath('{}_timing_curve.csv'.format(data_name), res_tgt)
    timedata.to_csv(timefile, index=False)

    # generate timing curve plot
    plt.figure()
    plt.title("Timing Curve ({})".format(data_name))
    plt.grid()
    plt.plot(train_sizes, train_time, marker='.', color='y', label='Train')
    plt.plot(train_sizes, predict_time, marker='.', color='dodgerblue', label='Predict')
    plt.legend(loc='best')
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('Elapsed user time in seconds')

    # save timing curve plot as PNG
    plotdir = 'plots'
    plot_tgt = '{}/{}'.format(plotdir, clf_name)
    plotpath = get_abspath('{}_TC.png'.format(data_name), plot_tgt)
    plt.savefig(plotpath)
    plt.close()


def create_iteration_curve(estimator, X_train, X_test, y_train, y_test, data_name, clf_name, param, scorer):
    """Generates an iteration curve for the specified estimator, saves tabular
    results to CSV and saves a plot of the iteration curve.

    Args:
        estimator (object): Target classifier.
        X_train (numpy.Array): Training features.
        X_test (numpy.Array): Test features.
        y_train (numpy.Array): Training labels.
        y_test (numpy.Array): Test labels.
        data_name (str): Name of data set being tested.
        clf_name (str): Type of algorithm.
        params (dict): Name of # iterations param for classifier.
        scorer (function): Scoring function.

    """
    # set variables
    iterations = np.arange(1, 300, 10)
    train_iter = []
    predict_iter = []
    final_df = []

    # start loop
    for i, iteration in enumerate(iterations):
        estimator.set_params(**{param: iteration})
        estimator.fit(X_train, y_train)
        train_iter.append(np.mean(cross_val_score(
            estimator, X_train, y_train, scoring=scorer, cv=4)))
        predict_iter.append(np.mean(cross_val_score(
            estimator, X_test, y_test, scoring=scorer, cv=4)))
        final_df.append([iteration, train_iter[i], predict_iter[i]])

    # save iteration results to CSV
    itercsv = pd.DataFrame(data=final_df, columns=[
        'Iterations', 'Train Accuracy', 'Test Accuracy'])
    resdir = 'results'
    res_tgt = '{}/{}'.format(resdir, clf_name)
    iterfile = get_abspath('{}_iterations.csv'.format(data_name), res_tgt)
    itercsv.to_csv(iterfile, index=False)

    # generate iteration curve plot
    plt.figure()
    plt.title("Iteration Curve ({})".format(data_name))
    plt.plot(iterations, train_iter, marker='.',
             color='m', label='Train Score')
    plt.plot(iterations, predict_iter, marker='.',
             color='c', label='Test Score')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('Number of iterations')
    plt.ylabel('Accuracy Score (Balanced)')
    plt.ylim((0, 1))

    # save iteration curve plot as PNG
    plotdir = 'plots'
    plot_tgt = '{}/{}'.format(plotdir, clf_name)
    plotpath = get_abspath('{}_IC.png'.format(data_name), plot_tgt)
    plt.savefig(plotpath)
    plt.close()


def create_validation_curve(estimator, X_train, y_train, data_name, clf_name, param_name, param_range, scorer):
    """Generates an validation/complexity curve for the ANN estimator, saves
    tabular results to CSV and saves a plot of the validation curve.

    Args:
        estimator (object): Target classifier.
        X_train (numpy.Array): Training features.
        y_train (numpy.Array): Training labels.
        data_name (str): Name of data set being tested.
        clf_name (str): Type of algorithm.
        param_name (dict): Name of parameter to be tested.
        param_range (dict): Range of parameter values to be tested.
        scorer (function): Scoring function.

    """
    # generate validation curve results
    train_scores, test_scores = validation_curve(
        estimator, X_train, y_train, param_name=param_name, param_range=param_range, cv=6, scoring=scorer, n_jobs=-1)

    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    
    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    
    # generate validation curve plot
    plt.figure()
    plt.title("Validation Curve ({})".format(data_name))
    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="darkorange", lw=2)
    plt.plot(param_range, test_mean, label="Cross-validation score", color="navy",lw=2)
    
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel(param_name)
    plt.ylabel('Accuracy Score')

    # save iteration curve plot as PNG
    plotdir = 'plots'
    plot_tgt = '{}/{}'.format(plotdir, clf_name)
    plotpath = get_abspath('{}_VC.png'.format(data_name), plot_tgt)
    plt.savefig(plotpath)
    plt.close()


if __name__ == '__main__':
    # remove existing combined_results.csv file
    try:
        combined = get_abspath('combined_results.csv', 'results')
        os.remove(combined)
    except:
        pass

    # set scoring function
    scorer = make_scorer(balanced_accuracy)

    # load datasets
    p_abalone = get_abspath('abalone.csv', 'data/experiments')
    p_abalone2 = get_abspath('abalone-2.csv', 'data/experiments')
    p_digits = get_abspath('digits.csv', 'data/experiments')
    
    df_abalone = pd.read_csv(p_abalone)
    df_abalone2 = pd.read_csv(p_abalone2)
    df_digits = pd.read_csv(p_digits)
    
    dfs = {'abalone2': df_abalone2, 'abalone': df_abalone, 'digits':df_digits}
    dnames = [  'abalone2']

    # instantiate dict of estimators
    estimators = {
#                  'KNN': None,
#                  'DT': None,
                  'ANN': None,
#                  'SVM_RBF': None,
#                  'SVM_PLY': None,
#                  'Boosting': None
                  }
    mnames = [
#            'KNN', 
##            'SVM_RBF', 
            'ANN', 
#            'DT', 
#            'SVM_PLY', 
##            'Boosting'
            ]

    # estimators with iteration param
    iterators = {'Boosting': 'ADA__n_estimators',
                 'ANN': 'MLP__max_iter'}

    # validation curve parameter names and ranges
    vc_params = {
                 'DT': ('DT__max_depth', np.arange(1, 50, 1)),
                 'KNN': ('KNN__n_neighbors', np.arange(1, 50, 1)),
                 'ANN': ('MLP__hidden_layer_sizes', np.arange(1,50,5)),
                 'SVM_RBF': ('SVMR__gamma',  np.logspace(-9, -1, 15)),
                 'SVM_PLY': ('SVMP__gamma',  np.logspace(-9, -1, 20)),
                 'Boosting': ('ADA__n_estimators', np.arange(20, 200, 10))
                 }
    
    # start model evaluation loop
    for df in dnames:
        X_train, X_test, y_train, y_test = split_data(dfs[df])
        # load pickled models into estimators dict
        for m in mnames:
            mfile = '{}\{}_grid.pkl'.format(m, df)
            model = load_pickled_model(get_abspath(mfile, filepath='models'))
            estimators[m] = model
       
        log_cols2=["Train Set  Accuracy", "Cross-Validation Score", "Test Set Accuracy"]
        log2 = pd.DataFrame(columns=log_cols2)   
        
        # generate validation, learning, and timing curves
        for name, estimator in estimators.iteritems():
    
            ### MODEL COMPLEXITY CURVES ###
#            basic_results(estimator, X_test, y_test, X_train, y_train, data_name=df, clf_name=name)
#            cv = 6
            plot_learning_curve(estimator.best_estimator_, name, X_train, y_train, scorer,  cv=cv, dn=df, n_jobs=4)
#            create_timing_curve(estimator.best_estimator_, dataset=dfs[df], data_name=df, clf_name=name)
#            create_validation_curve(estimator.best_estimator_, X_train, y_train, data_name=df, clf_name=name, param_name=vc_params[name][0], param_range=vc_params[name][1], scorer="accuracy")
##
#            #generate iteration curves for ANN and AdaBoost classifiers
#            if name == 'ANN' or name == 'Boosting':
#                create_iteration_curve(estimator.best_estimator_, X_train, X_test, y_train, y_test, data_name=df, clf_name=name, param=iterators[name], scorer=scorer)

            ### PERFORMACE CURVES ####
            be = estimator.best_estimator_
            print ('\nClassifier: {}'.format(name))
            print ('Data Name: {}'.format(df))
            print ('Best Estimator: {}'.format(be))
            be.fit(X_train, y_train)
            pred = be.predict(X_test)
            pred_tr = be.predict(X_train)
            print("Accuracy Score:{}".format(metrics.accuracy_score(y_true=y_test, y_pred=pred)*100))
#            print(("Error - {0}".format(1 - metrics.accuracy_score(y_true=y_test, y_pred=pred))))              
            print((metrics.confusion_matrix(y_test, pred)))
#            print("training")
            print((metrics.classification_report(y_train, pred_tr)))
#            print("testing")
#            print((metrics.classification_report(y_test, pred)))
            
#            acc = metrics.accuracy_score(y_true=y_test, y_pred=pred)
#            acc_tr = metrics.accuracy_score(y_true=y_train, y_pred=pred_tr)
#            scores = cross_val_score(be, X_train, y_train, cv=5)
#            log_entry = pd.DataFrame([[round(acc_tr*100,2) , round(scores.mean()*100,2), round(acc*100,2)]], columns=log_cols2)
#            log2 = log2.append(log_entry)
            
#        log2.index = ['KNN','ANN', 'SVC (RBF)', 'ADA', 'DT', 'SVC(POLY)']
#        ax = log2.plot.bar(rot=0,  width=0.8, figsize=(12,8))
#        for p in ax.patches:
#            ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
#        plt.xlabel('Accuracy %')
#        plt.title('Classifier Accuracy ({})'.format(df))
#        plt.legend(loc='lower right')
#        # save iteration curve plot as PNG
#        plotdir = 'plots/datasets'
#        plotpath = get_abspath('{}_acc.png'.format(df), plotdir)
#        plt.savefig(plotpath)
#        plt.close()
