import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np 
import matplotlib.pyplot as plt
import pylab
import sys
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

plt.rcParams["figure.figsize"] = [18.0, 8.0]

def read_data(file_name):
    '''
    Read in data and return a pandas df
    '''
    return pd.read_csv(file_name, header=0)

def print_statistics(data):
    '''
    Print statistics, correlation
    '''
    pd.set_option('display.width', 20)
    print '**** column names:  ', "\n", data.columns.values
    print '**** top of the data: ', "\n",  data.head()
    print '**** dataframe shape: ', "\n", data.shape
    print '**** statistics: ', "\n", data.describe(include='all')
    print '**** MODE: ', "\n", data.mode()
    print '**** sum of null values by column: ', "\n", data.isnull().sum()
    print '**** correlation matrix: ', "\n", data.corr()

def print_value_counts(data, col):
    '''
    For a given column in the data, print the counts 
    of the column's values.
    '''
    print pd.value_counts(data[col])

def visualize(data):
    data.hist()
    plt.savefig('hist.png')

def impute_missing_all(data, method):
    '''
    Find all columns with missing data and impute with the column's 
    mean, median, or mode.

    To impute specific columns, use impute_missing_column.
    '''

    headers = list(data.columns)
    for name in headers:
        if data[name].isnull().values.any():
            data[name] = data[name].fillna(data[name].method())

    assert not data.isnull().values.any()

def impute_missing_column(data, columns, method):
    '''
    Given a list of specific data columns, impute missing
    data of those columns with the column's mean, median, or mode.

    This function imputes specific columsn, for imputing all
    columns of the dataset that have missing data, use
    impute_missing_all.
    '''

    for col in columns:
        if method == 'median':
            data[col] = data[col].fillna(data[col].median())
        elif method == 'mode':
            data[col] = data[col].fillna(int(data[col].mode()[0]))
        else:
            data[col] = data[col].fillna(data[col].mean())

def log_column(data, column):
    '''
    Log the values of a column.

    Good to use when working with income data
    '''

    log_col = 'log_' + str(column)
    data[log_col] = data[column].apply(lambda x: np.log(x + 1))

def generate_categorical_features(data):
    pass
    #df.newcolumn = 1
    #https://github.com/yhat/DataGotham2013/blob/master/notebooks/7%20-%20Feature%20Engineering.ipynb

def generate_cont_features(data):
    pass

def model_data(training_data, test_data, features, label):
    
    model = LogisticRegression()
    model.fit(training_data[features], training_data[label])

    #return model 

    predicted = model.predict(test_data[features])
    print predicted
    #model.score(test_data[features], test_data[label])
    #print accuracy_score(predicted, test_data[label]) 

def evaluate_model(test_data, model):
    pass

def go(training_file, testing_file):
    '''
    run file
    '''
    df = read_data(training_file)

    #print_statistics(df)
    #visualize(df)

    # I'm imputting num dependents with 0 bc its the mode
    impute_missing_column(df, ['NumberOfDependents'], 'mode')

    # impute MonthlyIncome with median
    impute_missing_column(df, ['MonthlyIncome'], 'mean')
    #df['MonthlyIncome'] = df['MonthlyIncome'].fillna(int(df['MonthlyIncome'].median()))

    print 'dataframe has no null values?:', not df.isnull().values.any()
    print df.isnull().sum()

    #log income
    log_column(df, 'MonthlyIncome')
    
    features = ['RevolvingUtilizationOfUnsecuredLines', 
                'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
                'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
                'NumberRealEstateLoansOrLines', 
                'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
    label = 'SeriousDlqin2yrs'


    # split train and test data
    train, test = train_test_split(df, test_size = 0.2)


    #print model_data(train, test, features, label)
    
    #accuracy = evaluate_model(test, model)
    #evaluate_model(training_file)

    #print_value_counts(train, 'MonthlyIncome')
    #print train['MonthlyIncome'].median()

    # scaling 0 to 1
    # convert binary male / female to 0 and 1
if __name__=="__main__":
    instructions = '''Usage: python workflow.py training_file, test_file'''

    if(len(sys.argv) != 3):
        print(instructions)
        sys.exit()

    training_file = sys.argv[1]
    testing_file = sys.argv[2]

    go(training_file, testing_file)

