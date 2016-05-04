import pandas as pd
import csv
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np 
import matplotlib.pyplot as plt
import pylab
import sys
import random
from sklearn import svm, ensemble
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import *
from sklearn.feature_selection import RFE
from sklearn.grid_search import ParameterGrid
from multiprocessing import Pool
from functools import partial
from time import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.rcParams["figure.figsize"] = [18.0, 8.0]


clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=0),
    'LR': LogisticRegression(random_state=0, n_jobs=-1),
    'SVM': svm.LinearSVC(random_state=0, dual= False),
    'NB': GaussianNB(),
    'DT': DecisionTreeClassifier(random_state = 0),
    'KNN': KNeighborsClassifier(n_jobs = -1),
    'GB': GradientBoostingClassifier(random_state = 0)

        }

grid = { 
'RF':{'n_estimators': [1,10,100], 'max_depth': [1,5,10,20,50,75], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,5]},
'NB' : {},
'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1], 'penalty': ['l1', 'l2']},
'GB': {'n_estimators': [1,10,100], 'learning_rate' : [0.001,0.01,0.05,0.1],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10]},
'KNN' :{'n_neighbors': [1, 3, 5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
       }

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
            median = data[col].median()
            data[col] = data[col].fillna(median)
            return median
        elif method == 'mode':
            mode = int(data[col].mode()[0])
            data[col] = data[col].fillna(mode)
            return mode 
        else:
            mean = data[col].mean()
            data[col] = data[col].fillna(mean)
            return mean 


def impute_col_with_val(data, columns, value):
    '''
    Given data, a list of columns, and a value, impute the missing data of 
    given column with the value.

    Good to use to impute test data with training data's mean, median, or mode

    '''
    for col in columns:
        data[col] = data[col].fillna(value)

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


def evaluate_model(test_data, label, predicted_values):
    '''
    Compare the label of the test data to predicted values
    and return accuracy, precision, recall, and f1 score.

    '''
    accuracy = accuracy_score(test_data[label], predicted_values) 
    precision = precision_score(test_data[label], predicted_values) 
    recall = recall_score(test_data[label], predicted_values) 
    # f1 calculation is F1 = 2 * (precision * recall) / (precision + recall)
    f1 = f1_score(test_data[label], predicted_values) 

    return accuracy, precision, recall, f1

def evaluate_model_threshold(test_data, label, predicted_values, threshold):
    '''
    Compare the label of the test data to predicted values
    and return an accuracy score.
    '''
    accuracy = accuracy_score(test_data[label], predicted_values) 
    precision = precision_score(test_data[label], predicted_values) 
    recall = recall_score(test_data[label], predicted_values) 
    # f1 calculation is F1 = 2 * (precision * recall) / (precision + recall)
    f1 = f1_score(test_data[label], predicted_values) 

    return accuracy, precision, recall, f1

def plot_precision_recall(y_true, y_prob, model_name, model_params):

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision_curve[:-1]
    recall = recall_curve[:-1]
    plt.clf()
    plt.plot(recall, precision, label='%s' % model_params)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title("Precision Recall Curve for %s" %model_name)
    plt.savefig(model_name)
    plt.legend(loc="lower right")
    #plt.show()

def plot_precision_recall_all_models(y_true, y_prob_dict):

    plt.clf()

    for model_name, y_prob in y_prob_dict.items():
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
        precision = precision_curve[:-1]
        recall = recall_curve[:-1]
        plt.plot(recall, precision, label='%s' %model_name)


    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title("Precision Recall Curves")
    plt.savefig("pr-recall.png")
    plt.legend(loc="lower right")
    #plt.show()


def go(training_file):
    '''
    Run functions for specific data file
    '''
    
    ####### EXPLORE AND VISUALIZE ALL DATA
    df = read_data(training_file)
    #print_statistics(df)
    #visualize_all(df)

    #visualize_by_group_mean(df, ['NumberOfDependents', 'SeriousDlqin2yrs'], 'NumberOfDependents')
    #visualize_by_group_mean(df, [age_bucket, "SeriousDlqin2yrs"], age_bucket)
    #visualize_by_group_mean(df, [income_bucket, "SeriousDlqin2yrs"], income_bucket)

    #####################################
    # split train and test data
    ''' K FOLD SPLIT '''
    #kf = KFold(len(df), n_folds=3)

    #for train, test in kfold: 
    #print test
    train, test = train_test_split(df, test_size = 0.2)

    ##### IMPUTING AND TRANSFORMING TRAINING DATA
    # impute TRAIN DATA dependents with mode
    train_dependents_mode = impute_missing_column(train, ['NumberOfDependents'], 'mode')

    # impute TRAIN DATA MonthlyIncome with median
    train_income_median = impute_missing_column(train, ['MonthlyIncome'], 'median')

    #log income
    new_log_col = log_column(train, 'MonthlyIncome')

    age_bins = [0] + range(20, 80, 5) + [120]
    age_bucket = create_bins(train, 'age', age_bins)

    income_bins = range(0, 10000, 1000) + [train['MonthlyIncome'].max()]
    income_bucket = create_bins(train, 'MonthlyIncome', income_bins)

    scaled_income = scale_column(train, 'MonthlyIncome')

    assert not train.isnull().values.any()
    #################################################

    ########## IMPUTE AND TRANSFORM TEST DATA
    impute_col_with_val(test, ['NumberOfDependents'], train_dependents_mode)
    impute_col_with_val(test, ['MonthlyIncome'], train_income_median)

    #log income
    new_log_col = log_column(test, 'MonthlyIncome')
    age_bucket = create_bins(test, 'age', age_bins)
    income_bucket = create_bins(test, 'MonthlyIncome', income_bins)
    scaled_income = scale_column(test, 'MonthlyIncome')

    assert not test.isnull().values.any()

    ###########################################

    ######## GET FEATURES TO MODEL
    features = ['RevolvingUtilizationOfUnsecuredLines', 
                'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
                'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
                'NumberRealEstateLoansOrLines', 
                'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

    features = features + [new_log_col] + [age_bucket] + [income_bucket] + [scaled_income]

    label = 'SeriousDlqin2yrs'

    models_to_run=['LR','NB','DT', 'RF', 'SVM', 'GB']
    #models_to_run = ['LR', 'NB']
    best_overall_model = ''
    best_overall_auc = 0
    best_overall_params = ''

    #use a dict to save the y_prob values of models for plotting
    y_prob_dict = {}

    # create results-table csv
    with open('results-table.csv', 'wb') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        w.writerow(['MODEL', 'PARAMETERS', 'ACCURACY', 'PRECISION', 'RECALL', 'AUC'])

        start_loop = time()
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            running_model = models_to_run[index]
            print 'STARTING MODELS FOR', running_model
            parameter_values = grid[running_model]

            top_intra_model_auc = 0
            top_intra_model_params = ''

            for p in ParameterGrid(parameter_values):
                clf.set_params(**p)

                print 'STARTING %s WITH PARAMETERS %s' % (running_model, clf)
                start = time()
                clf.fit(train[features], train[label])
                elapsed_time = time() - start
                print '%s took %s seconds to fit' % (clf, elapsed_time)

                start = time()
                predicted_values = clf.predict(test[features])
                if hasattr(clf, 'predict_proba'):
                    y_pred_probs = clf.predict_proba(test[features])[:,1] #second col only for class = 1
                else:
                    y_pred_probs = clf.decision_function(test[features])

                elapsed_time = time() - start
                print '%s took %s seconds to get predicted values and proba' % (clf, elapsed_time)

                accuracy, precision, recall, f1 = evaluate_model(test, label, predicted_values)
                print 'METRICS USING PREDICTED VALUES:   ACCURACY: %s, PRECISION: %s, REACLL: %s, F1: %s' % (accuracy, precision, recall, f1)

                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(test[label], y_pred_probs)
                precision = precision_curve[:-1]
                recall = recall_curve[:-1]

                AUC = auc(recall, precision)
                print 'AUC SCORE', AUC

                # find best parameters within a model and its y_pred to plot
                if AUC > top_intra_model_auc:
                    top_intra_model_auc = AUC
                    top_intra_model_params = clf
                    top_intra_model_y_pred = y_pred_probs
                    y_prob_dict[running_model] = top_intra_model_y_pred

                # find best model and params overall
                if AUC > best_overall_auc:
                    best_overall_auc = AUC
                    best_overall_model = running_model
                    best_overall_params = clf


                print 'ENDED %s \n WITH PARAMETERS %s' % (running_model, clf)
                w.writerow([running_model, clf, accuracy, precision, recall, AUC])
            print 'ENDED MODELING FOR', running_model
         
        loop_time_minutes = (time() - start_loop) / 60
        print 'LOOP THRU ALL MODELS TOOK %s MINUTES' % loop_time_minutes
        print 'BEST MODEL %s \n BEST PARAMS %s \n BEST AUC %s \n' % (best_overall_model, best_overall_params, best_overall_auc)

        #report AUC for each fold, stdev
    # plot precision-recall curve for all models (picking the best parameters of each model)
    plot_precision_recall_all_models(test[label], y_prob_dict)

    #get best kfold values, get avg AUC

if __name__=="__main__":
    instructions = '''Usage: python workflow.py training_file'''

    if(len(sys.argv) != 2):
        print(instructions)
        sys.exit()

    training_file = sys.argv[1]

    go(training_file)

