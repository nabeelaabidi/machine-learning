# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 21:51:10 2018

@author: nabeela_zain
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
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


classifiers = [
   
    MLPClassifier(hidden_layer_sizes=(31,),
              learning_rate='constant', 
              max_iter=3000, solver='adam')
   
    ]
  # load datasets

p_abalone2 = get_abspath('abalone-2.csv', 'data/experiments')


log_cols=["Train Set  Accuracy", "Cross-Validation Score", "Test Set Accuracy"]
log = pd.DataFrame(columns=log_cols)


df_abalone2 = pd.read_csv(p_abalone2)

dfs = {'abalone2': df_abalone2}
dnames = ['abalone2']
for df in dnames:
        X_train, X_test, y_train, y_test = split_data(dfs[df])
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        for clf in classifiers:
            clf.fit(X_train, y_train)
            name = clf.__class__.__name__
            
            print("="*30)
            print(name)
            
            print('****Results****')
            train_predictions = clf.predict(X_test)
            pred_tr = clf.predict(X_train)
            acc = accuracy_score(y_test, train_predictions)
            acc_tr = accuracy_score(y_train, pred_tr)
            scores = cross_val_score(clf, X_train, y_train, cv=5)
            print("Accuracy: {:.4%}".format(acc))
            print clf.get_params()
     
            log_entry = pd.DataFrame([[round(acc_tr*100,2) , round(scores.mean()*100,2), round(acc*100,2)]], columns=log_cols)
            log = log.append(log_entry)
        print("="*30)
 
log.index = ['KNN', 'ANN', 'SVC (RBF)', 'ADA', 'DT', 'SVC(POLY)']

#ax = log.plot.bar(rot=0,  width=0.8, figsize=(12,8))
#
#for p in ax.patches:
#    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
#
#outpath='plots/datasets'
#plt.xlabel('Accuracy %')
#plt.title('Classifier Accuracy (Abalone)')
#plt.legend(loc='lower right')
#plt.show()
#plt.savefig(get_abspath('abalone_bar.png', outpath))
