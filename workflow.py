import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pylab
import requests
import json
import sys
from sklearn.linear_model import LogisticRegression

plt.rcParams["figure.figsize"] = [18.0, 8.0]
# plt.style.use('ggplot')

def read_data(file):
    '''
    Read in data and return a pandas df
    '''
    return pd.read_csv(file_name, header=0)

def print_statistics(data):
    '''
    Print statistics, correlation
    '''
    pd.set_option('display.width', 20)
    print '****column names:  ', "\n", data.columns.values
    print '****top of the data: ', "\n",  data.head()
    print '****dataframe shape: ', "\n", data.shape
    print '****statistics: ', "\n", data.describe(include='all')
    print '****sum of null values by column: ', "\n", data.isnull().sum()
    print '****correlation matrix: ', "\n", data.corr()

def visualize(data):
    data.hist()
    plt.savefig('hist.jpg')

def impute_missing(data):
    '''
    Find columns with missing data and impute with the column's mean.
    '''

    headers = list(data.columns)
    for name in headers:
        if data[name].isnull().values.any():
            data[name] = data[name].fillna(data[name].mean())

    assert not data.isnull().values.any()

def generate_categorical_features(data):
    pass

def generate_cont_features(data):
    pass

def model_data(data, X, y):
    
    model = LinearRegression()
    model.fit(X, y)


def evaluate_model(data):
    pass

def go(file_name):
    '''
    run file
    '''
    df = read_data(file_name)
    # print_statistics(df)
    # visualize(df)
    impute_missing(df)


    # X = ['Age', 'GPA', 'Days_missed']
    # y = 'Gender'
    #model_data(data, X, y)
    #create train / test data

if __name__=="__main__":
    instructions = '''Usage: python workflow.py filename'''

    if(len(sys.argv) != 2):
        print(instructions)
        sys.exit()

    file_name = sys.argv[1]
    go(file_name)

