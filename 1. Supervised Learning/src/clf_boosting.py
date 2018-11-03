from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np


class ADA(object):

    def __init__(self):
        """ Construct the AdaBoost classifier object

        """

        # set up model pipeline, scaling training data to have zero mean and
        # unit variance
        self.pipeline = Pipeline(
            [('Scale', StandardScaler()), ('ADA', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=0))])

        # set up parameter grid for parameters to search over
        self.params = {'ADA__learning_rate': np.arange(0.001, 0.2, 0.02),
                       'ADA__n_estimators': [10, 50, 100, 200],
                       'ADA__base_estimator__class_weight': [None],
                       'ADA__base_estimator__max_depth': np.arange(1, 20, 2),
                       'ADA__base_estimator__criterion': ['gini'],
                       }
