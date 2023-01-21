# Importing Dependencies
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
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

# Defining a custom Imputer class
class CustomImputer(BaseEstimator, TransformerMixin):
    # Initializing the objects
    def __init__(self, missing_values = np.nan, strategy = "mean", fill_value = None):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        
    # Defining the fit function
    def fit(self, X, y = None):
        # Creating a list of store an imputation value
        imputation_values = []
        
        # Creating a copy of a data frame
        data_frame = X.copy()
        
        # Creating a list of columns
        columns = data_frame.columns.tolist()
        
        # Looping through each column
        for i in columns:
            # Creating a condition based on imputation strategy
            if self.strategy == "mean":
                # Calculating the mean of a variable
                imputation_value = data_frame[i].mean()        
            elif self.strategy == "median":
                # Calculating the median of a variable
                imputation_value = data_frame[i].median()
            elif self.strategy == "constant":
                # Creating a condition based on fill_value parameter
                if self.fill_value is not None:
                    # Assigning the fill value to a new variable
                    imputation_value = self.fill_value
                else:
                    # Creating a condition based on data type of variable
                    if data_frame[i].dtype in ["object", "category"]:
                        # Redefining the imputation value as "missing_value"
                        imputation_value = "missing_value"
                    else:
                        # Redefining the imputation value as zero
                        imputation_value = 0
            else:
                # Calculating most frequent (mode) value of a variable
                imputation_value = data_frame[i].mode().values[0]
            
            # Appending the imputation value to the list
            imputation_values.append(imputation_value)
            
        # Redefining the objects
        self.imputation_values = imputation_values
        self.columns = columns
        
        # Returning the objects
        return self
    
    # Defining the transform function
    def transform(self, X, y = None):
        # Creating a copy of a data frame
        data_frame = X.copy()
        
        # Replacing the missing value indicators
        data_frame.replace(to_replace = self.missing_values, value = np.nan, inplace = True)
        
        # Looping through each column
        for index, column in enumerate(iterable = self.columns):
            # Imputing the missing values
            data_frame[column].fillna(value = self.imputation_values[index], inplace = True)
            
            # Asserting the number of missing values to be equal to zero
            assert data_frame.loc[data_frame[column].isna()].shape[0] == 0
        
        # Returning the data frame
        return data_frame

# Defining a custom Winsorizor class
class Winsorizor(BaseEstimator, TransformerMixin):
    # Initializing the objects
    def __init__(self, capping_method = "iqr", tail = "both", weight = 1.5, map_to_zero = False):
        self.capping_method = capping_method
        self.tail = tail
        self.weight = weight
        self.map_to_zero = map_to_zero
    
    # Defining the fit function
    def fit(self, X, y = None):
        # Creating an empty list to store upper boundary for a variable
        upper_boundaries = []
        
        # Creating an empty list to store lower boundary for a variable
        lower_boundaries = []
        
        # Creating a copy of a data frame
        data_frame = X.copy()
        
        # Creating a list of columns
        columns = data_frame.columns.tolist()
        
        # Looping through each column
        for i in columns:
            # Creating a condition based on winsorization method
            if self.capping_method == "iqr":
                # Calculating the third quantile of a variable
                Q3 = data_frame[i].quantile(q = 0.75)
                
                # Calculating the third quantile of a variable
                Q1 = data_frame[i].quantile(q = 0.25)
                
                # Calculating the inter quantile range method
                IQR = Q3 - Q1
                
                # Calculating the outlier range
                outlier_range = IQR * self.weight
                
                # Calculating the upper boundary
                upper_boundary = Q3 + outlier_range
                
                # Calculating the lower boundary
                lower_boundary = Q1 - outlier_range
                
                # Creating a condition based on mapping negative values as zero
                if not self.map_to_zero:
                    # Passing in case the condition is not satisfied
                    pass
                else:
                    # Creating a condition based on negative lower boundary value
                    if lower_boundary < 0:
                        # Redefining the lower boundary as zero
                        lower_boundary = 0
                    else:
                        pass
                    
            elif self.capping_method == "gaussian":
                # Calculating the mean of a variable
                mean = data_frame[i].mean()
                
                # Calculating the standard deviation of a variable
                std = data_frame[i].std()
                
                # Calculating the upper boundary
                upper_boundary = mean + (3 * std)
                
                # Calculating the lower boundary
                lower_boundary = mean - (3 * std)
                
                # Creating a condition based on mapping negative values as zero
                if not self.map_to_zero:
                    # Passing in case the condition is not satisfied
                    pass
                else:
                    # Creating a condition based on negative lower boundary value
                    if lower_boundary < 0:
                        # Redefining the lower boundary as zero
                        lower_boundary = 0
                    else:
                        pass
                    
            else:
                # Calculating the upper boundary
                upper_boundary = data_frame[i].quantile(q = 0.95)
                
                # Calculating the lower boundary
                lower_boundary = data_frame[i].quantile(q = 0.05)
                
                # Creating a condition based on mapping negative values as zero
                if not self.map_to_zero:
                    # Passing in case the condition is not satisfied
                    pass
                else:
                    # Creating a condition based on negative lower boundary value
                    if lower_boundary < 0:
                        # Redefining the lower boundary as zero
                        lower_boundary = 0
                    else:
                        pass
                
            # Appending the upper boundary to the list
            upper_boundaries.append(upper_boundary)
            
            # Appending the lower boundary to the list
            lower_boundaries.append(lower_boundary)
            
        # Redefining the objects
        self.upper_boundaries = upper_boundaries
        self.lower_boundaries = lower_boundaries
        self.columns = columns
            
        # Returning the objects
        return self
    
    # Defining the transform function
    def transform(self, X, y = None):
        # Creating a copy of a data frame
        data_frame = X.copy()
        
        # Looping through each column
        for index, column in enumerate(iterable = self.columns):
            # Creating a condition based on the tails of the distribution
            if self.tail == "both":
                # Winsorizing the outliers
                data_frame[column] = np.where(data_frame[column] > self.upper_boundaries[index], self.upper_boundaries[index], data_frame[column])
                data_frame[column] = np.where(data_frame[column] < self.lower_boundaries[index], self.lower_boundaries[index], data_frame[column])
            elif self.tail == "right":
                # Winsorizing the outliers
                data_frame[column] = np.where(data_frame[column] > self.upper_boundaries[index], self.upper_boundaries[index], data_frame[column])
            else:
                # Winsorizing the outliers
                data_frame[column] = np.where(data_frame[column] < self.lower_boundaries[index], self.lower_boundaries[index], data_frame[column])
        
        # Returning the data frame
        return data_frame
    
# Defining a custom VIFDropper class
class VIFDropper(BaseEstimator, TransformerMixin):
    # Initializing the object
    def __init__(self, vif_threshold = 2.5):
        self.vif_threshold = vif_threshold
        
    # Defining the fit function
    def fit(self, X, y = None):
        # Creating an empty list to store multicollinear features
        multicollinear_features = []
        
        # Creating a copy of a data frame
        data_frame = X.copy()
        
        # Creating a temporary data frame
        vif_df = pd.DataFrame()
        
        # Storing the features in the data frame
        vif_df["feature"] = data_frame.columns
        
        # Calculating the variance inflation factor (VIF) value for each feature
        vif_df["vif_value"] = [VIF(exog = data_frame.values, exog_idx = i) for i in range(len(data_frame.columns))]
        
        # Creating a condition based on the VIF threshold
        while vif_df.vif_value.max() >= self.vif_threshold:
            # Extracting the name of the feature with the highest VIF value
            multicollinear_feature = vif_df.loc[vif_df.vif_value == vif_df.vif_value.max(), "feature"].values[0]
            
            # Dropping the multicollinear feature from the data frame
            data_frame.drop(columns = multicollinear_feature, inplace = True)
            
            # Creating a temporary data frame
            vif_df = pd.DataFrame()

            # Storing the features in the data frame
            vif_df["feature"] = data_frame.columns

            # Calculating the variance inflation factor (VIF) value for each feature
            vif_df["vif_value"] = [VIF(exog = data_frame.values, exog_idx = i) for i in range(len(data_frame.columns))]
            
            # Appending the multicollinear feature to the list
            multicollinear_features.append(multicollinear_feature)
            
        # Redefining the object
        self.multicollinear_features = multicollinear_features
        
        # Returning the object
        return self
    
    # Defining the transform function
    def transform(self, X, y = None):
        # Creating a copy of a data frame
        data_frame = X.copy()
        
        # Dropping the multicollinear features
        data_frame.drop(columns = self.multicollinear_features, inplace = True)
        
        # Returning the data frame
        return data_frame
    
# Defining a custom class to apply Phi coefficient test
class PhiCoefficientDropper(BaseEstimator, TransformerMixin):
    """
    This is a custom class that will drop both of the binary features
    in case they are highly correlated based on the Phi coefficient.
    
    Arguments:
        threshold: A threshold to drop highly associated binary features.
    """
    # Defining the instance atributes
    def __init__(self, threshold = 0.6):
        self.threshold = threshold
        
    # Defining the fit method
    def fit(self, X, y = None):
        # Creating a copy of data frame
        copied_df = X.copy()
        
        # Creating a list of features
        features = copied_df.columns.tolist()
        
        # Creating an empty list
        phi_data = []
        
        # Looping through each feature
        for binary_var_1 in features:
            # Creating an empty list
            phi_coeffs = []
            
            # Looping through each feature
            for binary_var_2 in features:
                # Creating a contingency table
                contingency_table = pd.crosstab(index = copied_df[binary_var_1], columns = copied_df[binary_var_2]).values
                
                # Extracting the number of true positives (TP)
                TP = contingency_table[1][1]
                
                # Extracting the number of true negatives (TN)
                TN = contingency_table[0][0]
                
                # Extracting the number of false positives (FP)
                FP = contingency_table[0][1]
                
                # Extracting the number of false negatives (FN)
                FN = contingency_table[1][0]
                
                # Calculating the Phi coefficient
                phi = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
                
                # Appending the phi coefficients to the list
                phi_coeffs.append(phi)
            
            # Appending the list of phi coefficients to the list
            phi_data.append(phi_coeffs)
            
        # Creating a Phi correlation coefficient matrix
        phi_matrix = pd.DataFrame(data = phi_data, columns = features, index = features)
        
        # Calculating the absolute values, unstacking the matrix and sorting the features in descending order based on phi coefficients
        phi_coeff_df = phi_matrix.abs().unstack().sort_values(ascending = False).reset_index()
        
        # Renaming the variables
        phi_coeff_df.columns = ["binary_var_1", "binary_var_2", "phi"]
        
        # Removing the observations with same variables
        phi_coeff_df = phi_coeff_df.loc[phi_coeff_df.binary_var_1 != phi_coeff_df.binary_var_2].reset_index(drop = True)
        
        # Creating a list of highly associated binary variables
        highly_associated_binary_features = phi_coeff_df.loc[phi_coeff_df.phi >= self.threshold, "binary_var_1"].unique().tolist()
        
        # Defining the instance attributes
        self.highly_associated_binary_features = highly_associated_binary_features
        self.phi_coeff_df = phi_coeff_df
        self.phi_matrix = phi_matrix
        
        # Returning the object
        return self
    
    # Defining the transform method
    def transform(self, X, y = None):
        # Creating a copy of data frame
        copied_df = X.copy()
        
        # Dropping the highly correlated binary features
        copied_df = copied_df.drop(columns = self.highly_associated_binary_features)
        
        # Returning the transformed data frame
        return copied_df