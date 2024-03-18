
# Part 1: Decision Trees with Categorical Attributes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import  precision_recall_fscore_support, accuracy_score, confusion_matrix

# Return a pandas dataframe with data set to be mined.
# data_file will be populated with a string 
# corresponding to a path to the adult.csv file.
def read_csv_1(data_file):
	df = pd.read_csv(data_file ,delimiter=',')
	df = df.drop(columns=['fnlwgt'], errors='ignore')
	return df

# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	return df.shape[0]

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	return df.columns.tolist()

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	df.replace(' ?', pd.NA, inplace=True)
	return df.isna().sum().sum()

# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	df.replace(' ?', pd.NA, inplace=True)
	return df.columns[df.isna().any()].tolist()

# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters (by rounding to the first decimal digit)
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 21.547%, then the function should return 21.6.
def bachelors_masters_percentage(df):
	percentage = df[df['education'].isin(['Bachelors', 'Masters'])].shape[0] / df.shape[0] * 100
	return percentage

# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	df_cleaned = df.replace(' ?', pd.NA).dropna()
	return df_cleaned

# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function's output should not contain the target attribute.
def one_hot_encoding(df):
	# First, we remove the target attribute to ensure it's not encoded
    df_features = df.drop(columns=['class'])  # Dropping 'fnlwgt' as it's not required

    # Perform one-hot encoding
    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    features_encoded = onehot_encoder.fit_transform(df_features)
    
    # Convert the array back to a dataframe with appropriate column names
    columns_encoded = onehot_encoder.get_feature_names_out(df_features.columns)
    df_encoded = pd.DataFrame(features_encoded, columns=columns_encoded, index=df_features.index)
    
    return df_encoded

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	# Isolate the target attribute
    target = df[['class']].copy()  # Use double brackets to keep it as a dataframe

    # Perform label encoding
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target['class'])
    
    # Convert the array back to a pandas series
    series_encoded = pd.Series(target_encoded, name='class', index=target.index)

    return series_encoded

# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train, y_train):
    print("Starting to train the Decision Tree classifier...")
    # Create a decision tree classifier and fit it to the training data
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    print("Finished training. Now making predictions...")
    # Use the trained classifier to predict the labels of the training data
    y_pred = dt.predict(X_train)
    print("Predictions complete.")
    # No need to convert to a pandas series, since we're not doing any manipulation that requires index
    return y_pred

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
		print(f'Accuracy:{accuracy_score(y_true, y_pred)* 100:.2f}%')
		training_error_rate = 1 - accuracy_score(y_true, y_pred)
		print(f'Training error rate: {training_error_rate * 100:.2f}%') 
		return training_error_rate
