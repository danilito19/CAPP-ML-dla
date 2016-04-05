import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pylab
import requests
import json
import sys
from sklearn.linear_model import LogisticRegression

plt.rcParams["figure.figsize"] = [18.0, 8.0]
plt.style.use('ggplot')

def read_data(file):
    '''
    Read in data and return a pandas df
    '''
    return pd.read_csv(file_name, header=0)

def print_statistics(data):
    '''
    Print statistics, correlation
    '''
    print '****column names:  ', data.columns.values
    print '****top of the data: ', data.head()
    print '****dataframe shape: ', data.shape
    print '****statistics: ', data.describe(include='all')
    print '****sum of null values by column: ', data.isnull().sum()
    print '****correlation matrix: ', data.corr()

def visualize(data):
    data.hist()
    pylab.show()

def model_data(data):
    pass

def evaluate_model(data):
    pass


if __name__=="__main__":
    instructions = '''Usage: python workflow.py filename '''


    if(len(sys.argv) != 2):
        print(instructions)
        sys.exit()


    file_name = sys.argv[1]

    try:
        df = read_data(file_name)
        print_statistics(df)
        visualize(df)
        #create train / test data

    #if output to file: output all to file

        
    except Exception, e:
        print('Error: ', e)
