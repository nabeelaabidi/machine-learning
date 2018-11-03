# Decision Tree Classification

# Importing the libraries
import numpy as np
import pandas as pd
from helpers import load_pickled_model, get_abspath
from model_train import split_data, balanced_accuracy, balanced_f1
from sklearn.metrics import make_scorer
from sklearn import metrics
from sklearn.model_selection import learning_curve,  cross_val_score, validation_curve
import timeit
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Importing the dataset
p_abalone = get_abspath('abalone.csv', 'data/experiments')
p_abalone2 = get_abspath('abalone-2.csv', 'data/experiments')
p_seismic = get_abspath('seismic-bumps.csv', 'data/experiments')
df_abalone = pd.read_csv(p_abalone)
df_abalone2 = pd.read_csv(p_abalone2)
df_seismic = pd.read_csv(p_seismic)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = split_data(df_abalone2)
#dataset = pd.read_csv('Social_Network_Ads.csv')
#X = dataset.iloc[:, [2, 3]].values
#y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
classifier = SVC(C=0.90000000000000002, cache_size=8000, class_weight='balanced', coef0=1,
  decision_function_shape='ovr', degree=3, gamma=0.59,
  kernel='poly', max_iter=40000, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy_score = metrics.accuracy_score(y_test, y_pred)

## Visualising the Training set results
#from matplotlib.colors import ListedColormap
#X_set, y_set = X_train, y_train
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('Decision Tree Classification (Training set)')
#plt.xlabel('Age')
#plt.ylabel('Estimated Salary')
#plt.legend()
#plt.show()
#
## Visualising the Test set results
#from matplotlib.colors import ListedColormap
#X_set, y_set = X_test, y_test
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('Decision Tree Classification (Test set)')
#plt.xlabel('Age')
#plt.ylabel('Estimated Salary')
#plt.legend()
#plt.show()