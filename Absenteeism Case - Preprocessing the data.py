import pandas as pd
from IPython.core.display_functions import display

raw_csv_data = pd.read_csv("Absenteeism-data.csv")

type(raw_csv_data)
raw_csv_data
# Eyeballed the data to check the data for errors
df = raw_csv_data.copy()

pd.options.display.max_columns = None
pd.options.display.max_rows = 50
df.info()
# This is the concise summary of the dataframe
df = df.drop(['ID'], axis=1)
# Dropped the ID because it won't improve my model.
df['Reason for Absence'].unique()
# These are the categorical variables that classify the employees' reasons for their absence.
len(df['Reason for Absence'].unique())
# Since counting in Programming starts from 0, a number between 0 and 28 must be missing.
sorted(df['Reason for Absence'].unique())
# The missing value is 20.
# There must be a meaning behind these categorical variables.

reason_columns = pd.get_dummies(df['Reason for Absence'])
reason_columns['check'] = reason_columns.sum(axis=1)
reason_columns['check'].sum(axis=0)
# To check that the individual was absent from work due to one particular reason only.
reason_columns['check'].unique()
# To verify that the 700 values return only 1, which further proves that there are no missing or incorrect values.
reason_columns = reason_columns.drop(['check'], axis=1)
# I dropped the check column since I won't need it any further in preprocessing.
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
# Since I only want to predict the specific KNOWN reason that induces an individual to be excessively absent from work, I decided to drop reason 0 to avoid multicollinearity and preserve the logic of my model.
reason_columns

df.columns.values
reason_columns.columns.values
df = df.drop(['Reason for Absence'], axis=1)
# Dropped 'Reason for Absence' to avoid the duplication of information in my dataset, which triggers multicollinearity.

reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)
# Grouped/classified these variables to re-organize them, based on the Feature descriptions of Absenteeism of employees.

df
df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)
# To add the grouped reasons 1 to 4.
df.columns.values

column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                'Daily Work Load Average', 'Body Mass Index', 'Education',
                'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']

df.columns = column_names
# Renamed the grouped reasons to add them into the data in a more meaningful way.
df.head()
column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4',
                          'Date', 'Transportation Expense', 'Distance to Work', 'Age',
                          'Daily Work Load Average', 'Body Mass Index', 'Education',
                          'Children', 'Pets', 'Absenteeism Time in Hours']
# Reordered my data here for easier visualization.

df = df[column_names_reordered]
df.head()
df_reason_mod = df.copy()
# Made another checkpoint to have more control over my data.

# ## Preprocessing the 'Date'
type(df_reason_mod['Date'][0])
df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format='%d/%m/%Y')
df_reason_mod['Date']
# Preprocessed the 'Date' column to avoid inconsistencies.
df_reason_mod.info()
df_reason_mod['Date'][0]

# ## Preprocessing the 'Month'
df_reason_mod['Date'][0].month
# 7 represents the 7th month of the year, July.
list_months = []
list_months
# Created a list where I could store the reordered months.

df_reason_mod.shape
# length and width of the data table.

for i in range(df_reason_mod.shape[0]):
    list_months.append(df_reason_mod['Date'][i].month)
# Here I stored/added the reordered months.
list_months
len(list_months)
df_reason_mod['Month Value'] = list_months
# Added the column Month Value in the preprocessed data 
df_reason_mod.head()

# ## Preprocessed the Day of the Week
df_reason_mod['Date'][699].weekday()


def date_to_weekday(date_value):
    return date_value.weekday()


df_reason_mod['Day of the Week'] = df_reason_mod['Date'].apply(date_to_weekday)
# Added the date of the week to the data, where 0 is equals to Monday and so on until Sunday, which is equal to 6.
df_reason_mod = df_reason_mod.drop(['Date'], axis=1)
df_reason_mod.head()
# I decided to drop the 'Date' column because I already have the 'Month' and 'Date' columns to avoid duplication of information, which will trigger multicollinearity.

df_reason_mod.columns.values
column_names_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week',
                    'Transportation Expense', 'Distance to Work', 'Age',
                    'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                    'Pets', 'Absenteeism Time in Hours']
df_reason_mod = df_reason_mod[column_names_upd]
df_reason_mod.head()
# Reordered the data to preprocess it in a more meaningful manner.

df_reason_date_mod = df_reason_mod.copy()
df_reason_date_mod
# Created another checkpoint.

# # Checked 'Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average', and 'BMI' columns

type(df_reason_date_mod['Transportation Expense'][0])
type(df_reason_date_mod['Distance to Work'][0])
type(df_reason_date_mod['Age'][0])
type(df_reason_date_mod['Daily Work Load Average'][0])
type(df_reason_date_mod['Body Mass Index'][0])
df_reason_date_mod.info()
# I wanted to make sure that values in 'Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average', and 'BMI' columns are integers, so I could perform my analysis.

display(df_reason_date_mod)

# ## Preprocessed the 'Education'
df_reason_date_mod['Education'].unique()
df_reason_date_mod['Education'].value_counts()
# Where 1 corresponds to high school, 2 to graduate, 3 to postgraduate, and 4 to a master or a doctor.
# Looking at the values below, it makes more sense to combine number 2, 3, and 4 into a single category in order for their quantities to be more relevant relative to that of number 1.
df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1: 0, 2: 1, 3: 1, 4: 1})
# Here I transformed 'Education' into a Dummy Variable using the .map method.
# I also used the Dictionary feature of Python.
df_reason_date_mod['Education'].unique()
df_reason_date_mod['Education'].value_counts()
# I have successfully combined numbers 2, 3, and 4 in a single category

df_preprocessed = df_reason_date_mod.copy()
df_preprocessed.head(10)
# This is the final checkpoint.
# As a data scientist, I want a more manual way of doing preprocessing because it gives me a higher level of control over my data.
df_preprocessed.to_csv('Absenteeism_preprocessed.csv', index=False)
# I have successfully preprocessed the data and saved it in a csv file.
