# Importing Dependencies
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config

import pandas as pd
import numpy as np
import warnings

# Ignoring potential warnings
warnings.filterwarnings(action = "ignore")

# Visualizing the pipeline
set_config(display = "diagram")

# Defining a global level seed
np.random.seed(seed = 42)

# Creating a custom FrequencyRatioEncoder class
class FrequencyRatioEncoder(BaseEstimator, TransformerMixin):
    # Initializing the objects
    def __init__(self, use_arbitrary_frequency_ratio = False, unknown_value_frequency_ratio = None):
        self.use_arbitrary_frequency_ratio = use_arbitrary_frequency_ratio
        self.unknown_value_frequency_ratio = unknown_value_frequency_ratio
    
    # Defining the fit function
    def fit(self, X, y = None):
        # Creating an empty list to store frequency ratios
        frequencies = []
        
        # Creating a copy of a data frame
        data_frame = X.copy()
        
        # Creating a list of columns
        columns = data_frame.columns.tolist()
        
        # Looping through each column
        for i in columns:
            # Calculating the frequency ratio of each unique value of a variable
            frequency_dict = data_frame[i].value_counts(normalize = True).to_dict()
            
            # Appending the dictionary to the list
            frequencies.append(frequency_dict)
            
        # Redefining the objects
        self.frequencies = frequencies
        self.columns = columns
        
        # Returning the objects
        return self
    
    # Defining the transform function
    def transform(self, X, y = None):
        # Creating a copy of a data frame
        data_frame = X.copy()
        
        # Reseting the index
        data_frame.reset_index(drop = True, inplace = True)
        
        # Looping through each column
        for index, column in enumerate(iterable = self.columns):
            # Replacing the unique values with a frequency ratio
            data_frame[column] = data_frame[column].apply(func = lambda x: self.frequencies[index].get(x))
            
            # Creating a condition based on unseen value in a dataset
            if data_frame[column].isna().sum() > 0:
                # Creating a condition based on arbitrary frequency ratio
                if not self.use_arbitrary_frequency_ratio:
                    # Calculating the penalized frequency ratio
                    frequency_ratio = min(list(self.frequencies[index].values())) / 2
                    
                    # Replacing the unseen value with the penalized frequency ratio
                    data_frame.loc[data_frame[column].isna(), column] = frequency_ratio
                else:
                    # Replacing the unseen value with the unknown value frequency ratio
                    data_frame.loc[data_frame[column].isna(), column] = self.unknown_value_frequency_ratio
            else:
                # Passing in case the condition is not satisfied
                pass
            
            # Asserting the number of mising values to be equal to zero
            assert data_frame[column].isna().sum() == 0
        
        # Returning the data frame
        return data_frame

# Defining a custom class to encode rare categories
class RareLabelEncoder(BaseEstimator, TransformerMixin):
    # Defining the instance attributes
    def __init__(self, tol = 0.1, n_categories = 10, new_category = "Other", unseen_category = "Unseen Category"):
        self.tol = tol
        self.n_categories = n_categories
        self.new_category = new_category
        self.unseen_category = unseen_category

    # Defining the fit function
    def fit(self, X, y = None):
        # Creating a copy of a data frame
        data_frame = X.copy()

        # Creating a list of columns
        columns = data_frame.columns.tolist()

        # Creating a list of features with high cardinality
        high_cardinality_features = [i for i in columns if data_frame[i].nunique() >= self.n_categories]

        # Creating a list of value frequency for each feature
        category_frequencies = [data_frame[i].value_counts(normalize = True) for i in high_cardinality_features]

        # Creating a list of rare categories for each feature
        rare_categories = [i.loc[i < self.tol].index.tolist() for i in category_frequencies]

        # Looping through each loop
        for index, column in enumerate(iterable = high_cardinality_features):
            # Mapping the rare categories
            data_frame.loc[data_frame[column].isin(values = rare_categories[index]), column] = f"{self.new_category}_{column}"

        # Creating a list of unique categories
        unique_categories_train = [data_frame[i].unique().tolist() for i in high_cardinality_features]

        # Redefining the instance attributes
        self.high_cardinality_features = high_cardinality_features
        self.unique_categories_train = unique_categories_train
        self.rare_categories = rare_categories

        # Returning the objects
        return self

    # Defining the fit function
    def transform(self, X, y = None):
        # Creating a copy of a data frame
        data_frame = X.copy()

        # Looping through each loop
        for index, column in enumerate(iterable = self.high_cardinality_features):
            # Creating a list of rare categories
            rare_categories = self.rare_categories[index]

            # Creating a list of unique values of a train set
            unique_categories_train = self.unique_categories_train[index]

            # Mapping the rare categories
            data_frame.loc[data_frame[column].isin(values = rare_categories), column] = f"{self.new_category}_{column}"

            # Creating a list of unique values of a test set
            unique_categories_test = data_frame[column].unique().tolist()

            # Identifying the potential unseen categories
            unseen_categories = [i for i in unique_categories_test if i not in unique_categories_train]

            # Creating a condition based on the number of unseen categories
            if len(unseen_categories) > 0:
                # Mapping the unseen categories
                data_frame.loc[data_frame[column].isin(values = unseen_categories), column] = self.unseen_category
            else:
                # Passing in case the condition is not satisfied
                pass

        # Returning the data frame
        return data_frame