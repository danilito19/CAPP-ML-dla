import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# A 1
file_name = 'mock_student_data.csv'
df = pd.read_csv(file_name, header=0)
# print df.mode()
# print df.isnull().sum()
print df.describe()
df.hist()

# A 2 infer gender based on name with www.genderize.io

#get array of names with missing gender
#put 10 at a time in loop
# url = 'https://api.genderize.io/?name[0]=peter&name[1]=lois&name[2]=stevie'.format(i, )

# r = requests.get(url)
# try:
#     r.raise_for_status()
# except requests.exceptions.HTTPError as e:
#     print 'provide a correct zipcode'
#     return "Error: " + str(e)

#get dict and write back to file orj ust write dict?

# A 3
'''
 fill in the missing values for Age, GPA, and Days_missed using the following approaches:
Fill in missing values with the mean of the values for that attribute
Fill in missing values with a class-conditional mean (where the class is whether they graduated or not).
Is there a better, more appropriate method for filling in the missing values? If yes, describe and implement it. 
'''
# def fill_missing
