from helpers import get_abspath, save_dataset
import pandas as pd
import numpy as np
from sklearn import datasets

def preprocess_abalone():
    """Cleans and generates abalone dataset for experiments as a
    CSV file. Uses one-hot encoding for categorical features.
    """

    sdir = 'data\\abalone'
    tdir = 'data\experiments'
    abalone_file = get_abspath('abalone.csv', sdir)

    column_names = ["sex", "length", "diameter", "height", "whole weight", 
                "shucked weight", "viscera weight", "shell weight", "rings"]
    data = pd.read_csv(abalone_file, names=column_names)
    print("Number of samples (Abalone): %d" % len(data))

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X = LabelEncoder()
    data.iloc[:, 0] = labelencoder_X.fit_transform(data.iloc[:, 0])
    onehotencoder = OneHotEncoder(categorical_features = [0])
    data = onehotencoder.fit_transform(data).toarray()
    data = pd.DataFrame(data, columns =["female", "infant", "male", "length", "diameter", "height", "whole weight", 
                "shucked weight", "viscera weight", "shell weight", "rings"] )
    
    for col in data.columns:
        if len(np.unique(data[col])) == 1:
            data.drop(columns=col, inplace=True)
    
    # FOR 3 CLASSES
    for index, row in data.iterrows():
        if (row['rings'] < 7 ):
            row['rings'] = 0
        elif (row['rings'] >= 7 and (row['rings'] <= 13)):
            row['rings'] = 1
        elif (row['rings'] > 13):
            row['rings'] = 2
# 
    data.rename(columns={'rings': 'class'}, inplace=True)
#    save_dataset(data, 'abalone-2.csv', sep=',', subdir=tdir) # for 30 classes comment out when using 3 classes
    save_dataset(data, 'abalone.csv', sep=',', subdir=tdir) # for 3 classes

def preprocess_digits():
    tdir = 'data\experiments'
    digits = datasets.load_digits()
    X = digits.data  
    y = digits.target
    data = pd.DataFrame(X)
    data['class'] = pd.DataFrame(y)
    data = data.sample(frac=1).reset_index(drop=True)
    print("Number of samples: %d" % len(data))
    save_dataset(data, 'digits.csv', sep=',', subdir=tdir)
   
if __name__ == '__main__':
 
    preprocess_abalone()
    preprocess_digits()
 
