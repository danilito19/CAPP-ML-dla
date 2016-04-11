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
from sklearn.preprocessing import StandardScaler

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

    Returns the name of the new column to include programmatically in features
    '''

    log_col = 'log_' + str(column)
    data[log_col] = data[column].apply(lambda x: np.log(x + 1))

    return log_col

def create_age_bins(data, column, bins):
    '''
    Given a continuous variable, create a new column in the dataframe
    that represents the bin in which the continuous variable falls into.


    '''
    new_col = 'bins_' + str(column)

    data[new_col] = pd.cut(data[column], bins=bins)
    #print pd.value_counts(data['age_group'])

    return new_col

def convert_to_binary(data, column, zero_string):
    '''
    Given a binary categorical variable, such as a gender column with
    male and female, convert data to 0 for zero_string and 1 otherwise

    Provide the string of the forthcoming 0 value, such as 'male', "MALE", or 'Male"
    '''

    data[column] = data[column].apply(lambda x: 0 if sex == zero_string else 1)

def scale_column(data, column):

    scaled_col = 'scaled_' + str(column)
    data[scaled_col] = StandardScaler().fit_transform(data[column])

    return scaled_col

def model_data(training_data, test_data, features, label):
    
    model = LogisticRegression()
    model.fit(training_data[features], training_data[label])
    predicted = model.predict(test_data[features])
    return predicted

def evaluate_model(test_data, label, predicted_values):
    return accuracy_score(predicted_values, test_data[label]) 


def go(training_file):
    '''
    run file
    '''
    df = read_data(training_file)

    #print_statistics(df)
    #visualize(df)

    '''
    put in func
    cols = ['NumberOfDependents', 'SeriousDlqin2yrs']
    age_means = df[cols].groupby("NumberOfDependents").mean()
    print age_means
    age_means.plot()
    plt.savefig('means.png')

    df[["age_bucket", "serious_dlqin2yrs"]].groupby("age_bucket").mean()

    '''

    # I'm imputting num dependents with 0 bc its the mode
    impute_missing_column(df, ['NumberOfDependents'], 'mode')

    # impute MonthlyIncome with median
    impute_missing_column(df, ['MonthlyIncome'], 'mean')
    #df['MonthlyIncome'] = df['MonthlyIncome'].fillna(int(df['MonthlyIncome'].median()))

    # print 'dataframe has no null values?:', not df.isnull().values.any()
    # print df.isnull().sum()

    #log income
    new_log_col = log_column(df, 'MonthlyIncome')


    mybins = [0] + range(20, 80, 5) + [120]
    age_bucket = create_age_bins(df, 'age', mybins)


    new_col = scale_column(df, 'MonthlyIncome')


    features = ['RevolvingUtilizationOfUnsecuredLines', 
                'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
                'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
                'NumberRealEstateLoansOrLines', 
                'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

    #features = features + [new_log_col] + [age_bucket]
    features = features + [new_log_col] + [new_col]

    label = 'SeriousDlqin2yrs'

    # split train and test data
    train, test = train_test_split(df, test_size = 0.2)

    predicted_values = model_data(train, test, features, label)
    print evaluate_model(test, label, predicted_values)


    # FIND BEST MIX OF FEATURES

    '''
    Your task is to train one or more models on the training data and generate delinquency scores for the test data. 
    '''
if __name__=="__main__":
    instructions = '''Usage: python workflow.py training_file'''

    if(len(sys.argv) != 2):
        print(instructions)
        sys.exit()

    training_file = sys.argv[1]

    go(training_file)

