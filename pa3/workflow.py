import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np 
import matplotlib.pyplot as plt
import pylab
import sys
import random
from sklearn.linear_model import LogisticRegression
from sklearn import tree, svm, naive_bayes, neighbors, ensemble
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from time import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.rcParams["figure.figsize"] = [18.0, 8.0]

def read_data(file_name):
    '''
    Read in data and return a pandas df
    '''
    return pd.read_csv(file_name, header=0)

def print_statistics(data):
    '''
    Given a pandas dataframe, print dataframe statistics, correlation, and missing data.
    '''
    pd.set_option('display.width', 20)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
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

def visualize_all(data):
    '''
    Given a pandas dataframe, save a figure of dataframe column plots.
    '''

    data.hist()
    plt.savefig('all_data_hist.png')

def visualize_by_group_mean(data, cols, group_by_col):

    '''
    Given a dataframe, an array of columns and a column to group by,
    generate a plot of these grouped columns with mean of the group
    '''

    data[cols].groupby(group_by_col).mean().plot()
    file_name = 'viz_by_' + group_by_col + '.png'
    plt.savefig(file_name)


def impute_missing_all(data):
    '''
    Find all columns with missing data and impute with the column's 
    mean.

    To impute specific columns, use impute_missing_column.
    '''

    headers = list(data.columns)
    for name in headers:
        if data[name].isnull().values.any():
            data[name] = data[name].fillna(data[name].mean())

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

    Returns the name of the new column to include programmatically in list of features
    '''

    log_col = 'log_' + str(column)
    data[log_col] = data[column].apply(lambda x: np.log(x + 1))

    return log_col

def create_bins(data, column, bins, verbose=False):
    '''
    Given a continuous variable, create a new column in the dataframe
    that represents the bin in which the continuous variable falls into.

    If verbose is True, print the value counts of each bin.

    Returns the name of the new column to include programmatically in list of features

    '''
    new_col = 'bins_' + str(column)

    data[new_col] = pd.cut(data[column], bins=bins, include_lowest=True, labels=False)

    if verbose:
        print pd.value_counts(data[new_col])

    return new_col

def convert_to_binary(data, column, zero_string):
    '''
    Given a binary categorical variable, such as a gender column with
    male and female, convert data to 0 for zero_string and 1 otherwise

    Provide the string of the forthcoming 0 value, such as 'male', "MALE", or 'Male"
    '''

    data[column] = data[column].apply(lambda x: 0 if sex == zero_string else 1)

def scale_column(data, column):
    '''
    Given data and a specific column, apply a scale transformation to the column

    Returns the name of the new column to include programmatically in list of features

    '''

    scaled_col = 'scaled_' + str(column)
    data[scaled_col] = StandardScaler().fit_transform(data[column])

    return scaled_col

def model_logistic(training_data, test_data, features, label):

    '''
    With training and testing data and the data's features and label, select the best
    features with recursive feature elimination method, then
    fit a logistic regression model and return predicted values on the test data
    and a list of the best features used.

    '''
    start = time()
    model = LogisticRegression()
    rfe = RFE(model)
    rfe = rfe.fit(training_data[features], training_data[label])
    predicted = rfe.predict(test_data[features])
    best_features = rfe.get_support(indices=True)
    elapsed_time = time() - start
    print 'logistic regression took %s seconds to fit' %elapsed_time
    return predicted, best_features

def model_decision_tree(training_data, test_data, features, label):

    start = time()
    cl = tree.DecisionTreeClassifier(max_depth=8)
    cl.fit(training_data[features], training_data[label])
    predicted = cl.predict(test_data[features])

    elapsed_time = time() - start
    print 'decision tree took %s seconds to fit' %elapsed_time

    return predicted

def model_svm_linear(training_data, test_data, features, label):

    start = time()
    cl = svm.LinearSVC()
    cl.fit(training_data[features], training_data[label])
    predicted = cl.predict(test_data[features])

    elapsed_time = time() - start
    print 'linear svm took %s seconds to fit' %elapsed_time

    return predicted

def model_random_forest(training_data, test_data, features, label):

    start = time()
    cl = ensemble.RandomForestClassifier(n_estimators=100, max_depth=8, criterion='entropy')
    cl.fit(training_data[features], training_data[label])
    predicted = cl.predict(test_data[features])

    elapsed_time = time() - start
    print 'random forest took %s seconds to fit' %elapsed_time

    return predicted

def evaluate_model(test_data, label, predicted_values):
    '''
    Compare the label of the test data to predicted values
    and return an accuracy score.
    '''
    return accuracy_score(predicted_values, test_data[label]) 

def go(training_file):
    '''
    Run functions for specific data file
    '''
    
    df = read_data(training_file)
    #print_statistics(df)
    #visualize_all(df)

    # impute dependents with mode
    impute_missing_column(df, ['NumberOfDependents'], 'mode')

    # impute MonthlyIncome with median
    impute_missing_column(df, ['MonthlyIncome'], 'mean')

    assert not df.isnull().values.any()

    #log income
    new_log_col = log_column(df, 'MonthlyIncome')


    age_bins = [0] + range(20, 80, 5) + [120]
    age_bucket = create_bins(df, 'age', age_bins)

    income_bins = range(0, 10000, 1000) + [df['MonthlyIncome'].max()]
    income_bucket = create_bins(df, 'MonthlyIncome', income_bins)

    #visualize_by_group_mean(df, ['NumberOfDependents', 'SeriousDlqin2yrs'], 'NumberOfDependents')
    #visualize_by_group_mean(df, [age_bucket, "SeriousDlqin2yrs"], age_bucket)
    #visualize_by_group_mean(df, [income_bucket, "SeriousDlqin2yrs"], income_bucket)

    scaled_income = scale_column(df, 'MonthlyIncome')

    features = ['RevolvingUtilizationOfUnsecuredLines', 
                'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
                'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
                'NumberRealEstateLoansOrLines', 
                'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

    features = features + [new_log_col] + [age_bucket] + [income_bucket] + [scaled_income]

    label = 'SeriousDlqin2yrs'

    assert not df.isnull().values.any()

    # split train and test data
    train, test = train_test_split(df, test_size = 0.2)

    predicted_values, best_features = model_logistic(train, test, features, label)
    print 'THE LOGISTIC MODEL ACCURACY SCORE IS:',  evaluate_model(test, label, predicted_values)
    print
    print 'MODEL WAS BUILT WITH FEATURES : ', [features[i] for i in best_features] 

    predicted_values = model_decision_tree(train, test, features, label)
    print 'THE DECISION TREE MODEL ACCURACY SCORE IS:',  evaluate_model(test, label, predicted_values)

    predicted_values = model_svm_linear(train, test, features, label)
    print 'THE LINEAR SVM MODEL ACCURACY SCORE IS:',  evaluate_model(test, label, predicted_values)

    predicted_values = model_random_forest(train, test, features, label)
    print 'THE LINEAR SVM MODEL ACCURACY SCORE IS:',  evaluate_model(test, label, predicted_values)

if __name__=="__main__":
    instructions = '''Usage: python workflow.py training_file'''

    if(len(sys.argv) != 2):
        print(instructions)
        sys.exit()

    training_file = sys.argv[1]

    go(training_file)

