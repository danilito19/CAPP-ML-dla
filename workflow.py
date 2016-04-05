import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import requests
import json

def read_data(file):
    '''
    Read in data and return a pandas df
    '''
    pass

def print_statistics(data):
    '''
    Print statistics, correlation
    '''
    pass

def visualize(data):
    pass

def model_data(data):
    pass

def evaluate_model(data):
    pass


if __name__=="__main__":
    instructions = '''Usage: workflow file name '''


    if(len(sys.argv) != 2):
        print(instructions)
        sys.exit()

    file_name = sys.argv[2]

    try:
        pass
        
    except Exception, e:
        print('Error: ', e)
