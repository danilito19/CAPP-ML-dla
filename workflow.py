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
    Given a pandas dataframe, print dataframe statistics, correlation, and missing data.
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

    data[new_col] = pd.cut(data[column], bins=bins)

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
    With training and testing data and the data's features and label,
    fit a logistic regression model and return predicted values on the test data.

    '''
    
    model = LogisticRegression()
    model.fit(training_data[features], training_data[label])
    predicted = model.predict(test_data[features])
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
    age_bucket = create_bins(df, 'age', mybins)

    visualize_by_group_mean(df, ['NumberOfDependents', 'SeriousDlqin2yrs'], 'NumberOfDependents')
    visualize_by_group_mean(df, [age_bucket, "SeriousDlqin2yrs"], age_bucket)



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

    predicted_values = model_logistic(train, test, features, label)
    print evaluate_model(test, label, predicted_values)


    # FIND BEST MIX OF FEATURES

    '''
    Your task is to train one or more models on the training data and generate delinquency scores for the test data. 
    
    to get bins to Number
    df['income_bins'] = pd.cut(df.monthly_income, bins=15, labels=False)

    '''
if __name__=="__main__":
    instructions = '''Usage: python workflow.py training_file'''

    if(len(sys.argv) != 2):
        print(instructions)
        sys.exit()

    training_file = sys.argv[1]

    go(training_file)

