import pandas as pd
import numpy as np

data_preprocessed = pd.read_csv('df_preprocessed.csv')
data_preprocessed.head()
data_preprocessed['Absenteeism Time in Hours'].median()
# I used the median as a cutoff line so that the dataset will be balanced (there will be roughly equal number of 0s and 1s for the logistic regression).
# The median is 3 Hours/

targets = np.where(data_preprocessed['Absenteeism Time in Hours'] >
                   data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)
# created targets for the Regression which must be categories, so I can tell whether an employee is 'being absent too much' or not
# If the employee has been absent for 4 hours or more (more than 3 hours), then they will be assigned the value '1', that is the equivalent of taking half a day off
targets

data_preprocessed['Excessive Absenteeism'] = targets
# Created a Series in the original data frame that will contain the targets for the regression
data_preprocessed.head()

targets.sum() / targets.shape[0]
# Checked if the dataset is balanced.

data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours', 'Day of the Week',
                                            'Daily Work Load Average', 'Distance to Work'], axis=1)
# I dropped the unnecessary variables after exploring the weights.
data_with_targets is data_preprocessed
# I checked whether I created a checkpoint
data_with_targets.head()
data_with_targets.shape
data_with_targets.iloc[:, :-1]
# I excluded 'Excessive Absenteeism' because that column contains my targets.

unscaled_inputs = data_with_targets.iloc[:, :-1]
# Created a checkpoint variable that will contain the inputs (everything without the targets)

# ## Standardizing the data
from sklearn.preprocessing import StandardScaler

# I decided to use the StandardScaler module because it has much more capabilities than the straightforward 'preprocessing' method
absenteeism_scaler = StandardScaler()
# Here I defined StandardScaler as an object.
from sklearn.base import BaseEstimator, TransformerMixin


# These are the libraries needed to create the Custom Scaler.
# Created the CustomScaler class below.

class CustomScaler(BaseEstimator, TransformerMixin):

    # Here I declared what I want a CustomScaler object to do.

    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        # Used Keywords Arguments here
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std

    # The fit method based on StandardScaler.

    def fit(self, X, y=None):
        self.scaler = StandardScaler(copy=self.copy, with_mean=self.with_mean, with_std=self.with_std)
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    # The transform method which does the actual scaling.

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        # This records the initial order of the columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        # This scales all chosen features when creating the instance of the class
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        # This declares a variable containing all information that was not scaled
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
        # This returns a data frame which contains all scaled features and all 'not scaled' features


unscaled_inputs.columns.values
# Here I checked the number of columns I got

columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Education']

columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]
# Here I created the columns to scale, and excluded the columns to omit, and used list comprehension to iterate over the list

absenteeism_scaler = CustomScaler(columns_to_scale)
# Here I declared a CustomScaler object that specifies the columns I want to scale.

absenteeism_scaler.fit(unscaled_inputs)
# Here I fit the data to find the internal parameters of a model that will be used to transform the data.

scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
scaled_inputs
# Transformed data.
scaled_inputs.shape
# I checked the shape of the inputs


# ## Split the data into train & test and shuffle
from sklearn.model_selection import train_test_split

train_test_split(scaled_inputs, targets)
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8,
                                                    test_size=0.2, random_state=20)
print(x_train.shape, y_train.shape)
# Checked the shape of the train inputs and targets
print(x_test.shape, y_test.shape)
# Checked the shape of the test inputs and targets


# ## Logistic regression with sklearn
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()
reg.fit(x_train, y_train)
reg.score(x_train, y_train)

model_outputs = reg.predict(x_train)
model_outputs

model_outputs == y_train
np.sum((model_outputs == y_train))
# This is the number of instances that the trained model predicted correctly.
model_outputs.shape[0]
np.sum((model_outputs == y_train)) / model_outputs.shape[0]
# I manually checked the accuracy of the model.

reg.intercept_
# Bias of the model
reg.coef_
# Weights of the model

unscaled_inputs.columns.values
# To check the names of the columns

feature_name = unscaled_inputs.columns.values
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
# I moved the intercept to the top of the summary table and added it at index 0, then sorted the dataframe by index.
summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)
# Created the Odds Ratio for each feature.
summary_table.sort_values('Odds_ratio', ascending=False)
summary_table
# Sorted the table by the Odds Ratio of each feature, in descending order.

reg.score(x_test, y_test)
predicted_proba = reg.predict_proba(x_test)
predicted_proba
# These are the predicted probabilities of each class. 
# The first column shows the probability of a particular observation to be 0, while the second one - to be 1
predicted_proba.shape
predicted_proba[:, 1]

import pickle

with open('model', 'wb') as file:
    pickle.dump(reg, file)
# Here I pickled the model file.
with open('scaler', 'wb') as file:
    pickle.dump(absenteeism_scaler, file)
# Here I pickled the scaler file.

from absenteeism_module import *

model = absenteeism_model('model', 'scaler')
model.load_and_clean_data('Absenteeism_new_data.csv')
model.predicted_outputs()
