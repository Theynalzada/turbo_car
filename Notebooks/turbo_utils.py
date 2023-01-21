# Importing Dependencies
from sklearn.feature_selection import RFECV, SelectFpr, f_regression, SelectFromModel, SelectPercentile, SequentialFeatureSelector
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler, MaxAbsScaler, StandardScaler, FunctionTransformer
from turbo_preprocessing import Winsorizor, VIFDropper, CustomImputer, PhiCoefficientDropper
from turbo_category_encoders import RareLabelEncoder, FrequencyRatioEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn import set_config
from boruta import BorutaPy

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import warnings
import skopt
import yaml
import os

# Configuring the font scale, style and palette for plots
sns.set(font_scale = 1.5, style = "darkgrid", palette = "bright")

# Filtering potential warnings
warnings.filterwarnings(action = "ignore")

# Visualizing the pipeline
set_config(display = "diagram")

# Creating a global level seed
np.random.seed(seed = 42)

# Loading the configuration file
with open(file = "/Users/kzeynalzade/Documents/Turbo Project/Configuration/config.yml") as yaml_file:
    config = yaml.safe_load(stream = yaml_file)

# Extracting the target path for the dataset
target = config.get("target")

# Extracting the target path for the dataset
target_path = config.get("target_path")

# Defining the real path
real_target_path = os.path.realpath(filename = target_path)

# Extracting the evaluation metric
evaluation_metric = config.get("metric")

# Extracting the currency exchange rates
usd_to_azn, eur_to_azn = config.get("currency_exchange")

# Extracting the origin of the brands
origins_dict = config.get("origin")

# Extracting the luxury brands
luxury_brands = config.get("luxury_brands")

# Extracting the high cardinality features
HIGH_CARDINALITY_FEATURES = config.get("features").get("high_cardinality_features")

# Extracting the numeric features
NUMERIC_FEATURES = config.get("features").get("numeric_features")

# Extracting the nominal features
NOMINAL_FEATURES = config.get("features").get("nominal_features")

# Extracting the binary features
BINARY_FEATURES = config.get("features").get("binary_features")

# Defining a function to convert currency to AZN
def convert_currency(dataset = None):
    """
    This function is used to convert the currency to AZN.
    
    Args:
        dataset: A pandas data frame.
        
    Returns:
        Pandas data frame.
    """
    # Converting the price in EURO to AZN
    dataset.loc[dataset.currency == "EUR", "target"] = dataset.loc[dataset.currency == "EUR", "target"] * eur_to_azn
    
    # Converting the price in USD to AZN
    dataset.loc[dataset.currency == "USD", "target"] = dataset.loc[dataset.currency == "USD", "target"] * usd_to_azn

    # Casting the data dtype of the target variable
    dataset.target = dataset.target.astype(dtype = "float32")
    
    # Dropping the currency variable
    dataset.drop(columns = "currency", inplace = True)
    
    # Returning the data frame
    return dataset

# Defining a function to load the dataset
def load_data(filepath = real_target_path):
    """
    This is a function to load the dataset.
    
    Args:
        filepath: A file path to the dataset.
        
    Returns:
        Pandas data frame.
    """
    # Reading the data from a parquet file 
    dataset = pd.read_parquet(path = filepath, engine = "fastparquet")
    
    # Asserting the number of duplicate observations to be equal to zero
    assert dataset.duplicated(subset = "ID").sum() == 0
    
    # Renaming the dependent variable
    dataset.rename(columns = {target: "target"}, inplace = True)
    
    # Creating a chain of functions to preprocess features and target
    dataset = convert_currency(dataset = dataset)
    
    # Returning the dataset
    return dataset
    
# Defining a custom DataTypeConverter class
class DataTypeConverter(BaseEstimator, TransformerMixin):
    # Initializing the objects
    def __init__(self, dtype = "float32"):
        self.dtype = dtype
    
    # Defining the fit function
    def fit(self, X, y = None):
        # Returning the array
        return self
    
    # Defining the transform function
    def transform(self, X, y = None):
        # Creating a dictionary of data types
        dtypes_dict = {"float16": np.float16,
                       "float32": np.float32,
                       "float64": np.float64,
                       "float128": np.float128}
        
        # Converting the data type of array
        converted_array = np.array(object = X, dtype = dtypes_dict.get(self.dtype))
        
        # Returning the data frame
        return converted_array
    
# Defining a custom class to remove constant features
class ConstantDropper(BaseEstimator, TransformerMixin):
    # Defining the fit function
    def fit(self, X, y = None):
        # Creating a copy of a data frame
        data_frame = pd.DataFrame(data = X).copy()
        
        # Creating a list of features that contain only a single unique value
        constant_features = [i for i in data_frame.columns.tolist() if data_frame[i].nunique() == 1]
        
        # Redefining the object
        self.constant_features = constant_features
        
        # Returning the object
        return self
    
    # Defining the transform function
    def transform(self, X, y = None):
        # Creating a copy of a data frame
        data_frame = pd.DataFrame(data = X).copy()
        
        # Dropping the constant features
        data_frame.drop(columns = self.constant_features, inplace = True)
        
        # Returning the transformed data frame
        return data_frame.values
    
# Defining a custom InitialPreprocessor class
class InitialPreprocessor(BaseEstimator, TransformerMixin):
    # Defining the fit method
    def fit(self, X, y = None):
        # Returning the data
        return self
    
    # Defining the transform method
    def transform(self, X, y = None):
        # Creating a copy of a data frame
        data_frame = X.copy()
        
        # Defining the current year
        current_year = datetime.date.today().year

        # Calculating the age of the vehicles
        data_frame.year = (current_year - data_frame.year).astype(dtype = "int32")

        # Renaming the year variable as age
        data_frame.rename(columns = {"year": "age"}, inplace = True)

        # Removing the strings from the values of engine, mileage, and hp variables
        data_frame[["engine", "mileage", "hp"]] = data_frame[["engine", "mileage", "hp"]].applymap(func = lambda x: x.split()[0])

        # Casting the data type of engine variable from object to float with 32 precision
        data_frame.engine = data_frame.engine.astype(dtype = "float32")

        # Casting the data type of engine variable from object to integer with 32 precision
        data_frame[["mileage", "hp"]] = data_frame[["mileage", "hp"]].astype(dtype = "int32")

        # Calculating the average mileage per year
        data_frame["avg_mileage_per_year"] = (data_frame.mileage / data_frame.age).fillna(value = 0).apply(func = lambda x: round(number = x, ndigits = 1))

        # Mapping the infinite values
        data_frame.avg_mileage_per_year = np.where(data_frame.avg_mileage_per_year == np.inf, data_frame.mileage, data_frame.avg_mileage_per_year)

        # Encoding the cars that have mileage more than 150 thousand km
        data_frame["is_overdriven"] = np.where(data_frame.mileage >= 150000, 1, 0)

        # Replacing the values of is_new variable with binary values
        data_frame.is_new = np.where(data_frame.is_new == "Bəli", 1, 0)

        # Replacing the values of exchange_available variable with binary values
        data_frame.exchange_available = np.where(data_frame.exchange_available == "Barter mümkündür", 1, 0)

        # Replacing the values of is_saloon_car variable with binary values
        data_frame.is_saloon_car = np.where(data_frame.is_saloon_car == "Salon", 1, 0)

        # Replacing the values of loan_available variable with binary values
        data_frame.loan_available = np.where(data_frame.loan_available == "Kreditdədir", 1, 0)

        # Parsing the values of the functionalities variable
        data_frame.functionalities = data_frame.functionalities.apply(func = lambda x: [i[1:-1] for i in x.strip("[]").split(", ")])

        # Calculating the number of functionalities
        data_frame["n_functionalities"] = data_frame.functionalities.apply(func = lambda x: len(x))

        # Calculating the number of ads in case no functionality is specified
        n_cases = data_frame.loc[data_frame.n_functionalities == 0].shape[0]

        # Creating a condition to check if no functionality is specified
        if n_cases > 0:
            # Removing the add in case no functionality is specified
            data_frame = data_frame.loc[data_frame.n_functionalities > 0].reset_index(drop = True)
        else:
            # Passing in case the condition is not satisfied
            pass

        # Creating a binary variable to identify whether or not a car has allow wheels
        data_frame["has_alloy_wheels"] = data_frame.functionalities.apply(func = lambda x: int("Yüngül lehimli disklər" in x))

        # Creating a binary variable to identify whether or not a car has Anti-lock braking (ABS) system
        data_frame["has_abs"] = data_frame.functionalities.apply(func = lambda x: int("ABS" in x))

        # Creating a binary variable to identify whether or not a car has a hatch
        data_frame["has_hatch"] = data_frame.functionalities.apply(func = lambda x: int("Lyuk" in x))

        # Creating a binary variable to identify whether or not a car has a rain sensor
        data_frame["has_rain_sensor"] = data_frame.functionalities.apply(func = lambda x: int("Yağış sensoru" in x))

        # Creating a binary variable to identify whether or not a car has a central locking
        data_frame["has_central_locking"] = data_frame.functionalities.apply(func = lambda x: int("Mərkəzi qapanma" in x))

        # Creating a binary variable to identify whether or not a car has a parking radar
        data_frame["has_parking_radar"] = data_frame.functionalities.apply(func = lambda x: int("Park radarı" in x))

        # Creating a binary variable to identify whether or not a car has an air conditioner
        data_frame["has_air_conditioner"] = data_frame.functionalities.apply(func = lambda x: int("Kondisioner" in x))

        # Creating a binary variable to identify whether or not a car has a seat heating
        data_frame["has_seat_heating"] = data_frame.functionalities.apply(func = lambda x: int("Oturacaqların isidilməsi" in x))

        # Creating a binary variable to identify whether or not a car has a leather salon
        data_frame["has_leather_salon"] = data_frame.functionalities.apply(func = lambda x: int("Dəri salon" in x))

        # Creating a binary variable to identify whether or not a car has xenon lamps
        data_frame["has_xenon_lamps"] = data_frame.functionalities.apply(func = lambda x: int("Ksenon lampalar" in x))

        # Creating a binary variable to identify whether or not a car has a rear view camera
        data_frame["has_rear_view_camera"] = data_frame.functionalities.apply(func = lambda x: int("Arxa görüntü kamerası" in x))

        # Creating a binary variable to identify whether or not a car has side curtains
        data_frame["has_side_curtains"] = data_frame.functionalities.apply(func = lambda x: int("Yan pərdələr" in x))

        # Creating a binary variable to identify whether or not a car has a seat ventilation
        data_frame["has_seat_ventilation"] = data_frame.functionalities.apply(func = lambda x: int("Oturacaqların ventilyasiyası" in x))

        # Creating a new variable called origin
        data_frame["origin"] = " "

        # Creating a list of unique origins
        origins = list(origins_dict.keys())

        # Looping through each origin
        for origin in origins:
            # Extracting the brands associated with the origin
            brands = origins_dict.get(origin)

            # Mapping the brands to a specific origin 
            data_frame.loc[data_frame.brand.isin(values = brands), "origin"] = origin.capitalize()

        # Creating a variable to indiciate whether a particular brand is luxury or not
        data_frame["is_luxury_brand"] = np.where(data_frame.brand.isin(values = luxury_brands), 1, 0)
        
        # Creating a list of reallocated columns
        reallocated_columns = NOMINAL_FEATURES + HIGH_CARDINALITY_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES

        # Reallocating the columns
        data_frame = data_frame[reallocated_columns]
        
        # Returning the data frame
        return data_frame
    
# Defining a function to build a regressor pipeline
def build_pipeline(regressor = None, 
                   train_features = None,
                   train_labels = None,
                   apply_log_transformation = False,
                   apply_phi_test = False,
                   drop_multicolliner_features = False,
                   winsorize_outliers = False,
                   apply_feature_scaling = False,
                   feature_scaler_type = None,
                   apply_feature_selection = False,
                   feature_selector_type = None,
                   tune_hyperparameters = False,
                   hyperparameters = None,
                   n_iterations = None,
                   metric = evaluation_metric,
                   verbosity = 0):
    """
    This function is used to build a regressor pipeline.
    
    Args:
        regressor: A regressor instance.
        train_features: Training features as pandas data frame.
        train_labels: Ground truth labels of a train set.
        apply_log_transformation: Whether or not to apply log transformation.
        drop_multicolliner_features: Whether or not to drop multicollinear features.
        winsorize_outliers: Whether or not to winsorize outliers.
        apply_feature_scaling: Whether or not to apply feature scaling.
        feature_scaler_type: The type of a feature scaling method.
        apply_feature_selection: Whether or not to apply feature selection.
        feature_selector_type: The type of a feature selection method.
        tune_hyperparameters: Whether or not to tune hyperparameters.
        hyperparameters: A dictionary of hyperparameters.
        n_iterations: The number of iterations.
        metric: An evaluation metric to optimize the regressor.
        verbosity: A level of verbosity to display an output.
    
    Returns:
        A regressor pipeline.
    """
    # Instantiating the cross validation technique
    kf = KFold(shuffle = True, random_state = 42)
    
    # Creating a dictionary of feature scalers
    metrics_dict = {"mape": "neg_mean_absolute_percentage_error",
                    "rmse": "neg_root_mean_squared_error",
                    "mae": "neg_mean_absolute_error",
                    "mse": "neg_mean_squared_error"}
    
    # Creating a dictionary of feature scalers
    feature_selectors_dict = {"wrapper": SequentialFeatureSelector(estimator = regressor, scoring = metrics_dict.get(metric), cv = kf, n_jobs = -1),
                              "tree": BorutaPy(estimator = regressor, random_state = 42),
                              "univariate": SelectPercentile(percentile = 50),
                              "p_value": SelectFpr(score_func = f_regression),
                              "meta": SelectFromModel(estimator = regressor)}
    
    # Creating a dictionary of feature scalers
    feature_scalers_dict = {"standard": StandardScaler(),
                            "robust": RobustScaler(),
                            "minmax": MinMaxScaler(),
                            "maxabs": MaxAbsScaler()}
    
    # Creating a list of high cardinality features
    high_cardinality_features = HIGH_CARDINALITY_FEATURES
    
    # Creating a list of nominal features
    nominal_features = NOMINAL_FEATURES
    
    # Creating a list of numeric features
    numeric_features = NUMERIC_FEATURES
    
    # Creating a list of binary features
    binary_features = BINARY_FEATURES
    
    # Creating a pipeline for nominal features
    nominal_pipeline = Pipeline(steps = [("imputer", CustomImputer(strategy = "most_frequent")),
                                         ("rle", RareLabelEncoder(n_categories = 6)),
                                         ("ohe", OneHotEncoder(handle_unknown = "ignore"))])
    
    # Creating a pipeline for high cardinalit features
    high_cardinality_pipeline = Pipeline(steps = [("imputer", CustomImputer(strategy = "most_frequent")),
                                                  ("fre", FrequencyRatioEncoder())])
    
    # Creating a condition to apply Phi coefficient test
    if not apply_phi_test:
        # Creating a pipeline for binary features
        binary_pipeline = Pipeline(steps = [("imputer", CustomImputer(strategy = "most_frequent"))])
    else:
        # Creating a pipeline for binary features
        binary_pipeline = Pipeline(steps = [("imputer", CustomImputer(strategy = "most_frequent")),
                                            ("phi_test", PhiCoefficientDropper())])
    
    # Creating a condition based on the preprocessing steps for numeric features
    if not drop_multicolliner_features and not winsorize_outliers and not apply_feature_scaling:
        # Creating a pipeline for numeric features
        numeric_pipeline = Pipeline(steps = [("imputer", CustomImputer(strategy = "median"))])
    elif drop_multicolliner_features and not winsorize_outliers and not apply_feature_scaling:
        # Creating a pipeline for numeric features
        numeric_pipeline = Pipeline(steps = [("imputer", CustomImputer(strategy = "median")),
                                             ("vif_dropper", VIFDropper())])
    elif not drop_multicolliner_features and winsorize_outliers and not apply_feature_scaling:
        # Creating a pipeline for numeric features
        numeric_pipeline = Pipeline(steps = [("imputer", CustomImputer(strategy = "median")),
                                             ("winsorizor", Winsorizor(map_to_zero = True))])
    elif not drop_multicolliner_features and not winsorize_outliers and apply_feature_scaling:
        # Creating a pipeline for numeric features
        numeric_pipeline = Pipeline(steps = [("imputer", CustomImputer(strategy = "median")),
                                             ("feature_scaler", feature_scalers_dict.get(feature_scaler_type))])
    elif drop_multicolliner_features and winsorize_outliers and not apply_feature_scaling:
        # Creating a pipeline for numeric features
        numeric_pipeline = Pipeline(steps = [("imputer", CustomImputer(strategy = "median")),
                                             ("vif_dropper", VIFDropper()),
                                             ("winsorizor", Winsorizor(map_to_zero = True))])
    elif drop_multicolliner_features and not winsorize_outliers and apply_feature_scaling:
        # Creating a pipeline for numeric features
        numeric_pipeline = Pipeline(steps = [("imputer", CustomImputer(strategy = "median")),
                                             ("vif_dropper", VIFDropper()),
                                             ("feature_scaler", feature_scalers_dict.get(feature_scaler_type))])
    elif not drop_multicolliner_features and winsorize_outliers and apply_feature_scaling:
        # Creating a pipeline for numeric features
        numeric_pipeline = Pipeline(steps = [("imputer", CustomImputer(strategy = "median")),
                                             ("winsorizor", Winsorizor(map_to_zero = True)),
                                             ("feature_scaler", feature_scalers_dict.get(feature_scaler_type))])
    else:
        # Creating a pipeline for numeric features
        numeric_pipeline = Pipeline(steps = [("imputer", CustomImputer(strategy = "median")),
                                             ("vif_dropper", VIFDropper()),
                                             ("winsorizor", Winsorizor(map_to_zero = True)),
                                             ("feature_scaler", feature_scalers_dict.get(feature_scaler_type))])
        
    # Creating a feature transformer
    feature_transformer = ColumnTransformer(transformers = [("nominal_pipeline", nominal_pipeline, nominal_features),
                                                            ("high_cardinality_pipeline", high_cardinality_pipeline, high_cardinality_features),
                                                            ("binary_pipeline", binary_pipeline, binary_features),
                                                            ("numeric_pipeline", numeric_pipeline, numeric_features)],
                                            n_jobs = -1)
    
    # Creating a condition based on the feature selection
    if not apply_feature_selection:
        # Creating a regressor pipeline
        pipe = Pipeline(steps = [("initial_preprocessor", InitialPreprocessor()),
                                 ("feature_transformer", feature_transformer),
                                 ("constant_feature_dropper", ConstantDropper()),
                                 ("dtype_converter", DataTypeConverter()),
                                 ("regressor", regressor)])
    else:
        # Creating a regressor pipeline with feature selector instance included
        pipe = Pipeline(steps = [("initial_preprocessor", InitialPreprocessor()),
                                 ("feature_transformer", feature_transformer),
                                 ("constant_feature_dropper", ConstantDropper()),
                                 ("dtype_converter", DataTypeConverter()),
                                 ("feature_selector", feature_selectors_dict.get(feature_selector_type)),
                                 ("regressor", regressor)])
        
    # Creating a condition based on log transformation
    if not apply_log_transformation:
        # Passing in case the condition is not satisfied
        pass
    else:
        # Applying log transformation to the target variable
        pipe = TransformedTargetRegressor(regressor = pipe, func = np.log, inverse_func = np.exp)
        
    # Creating a condition based on hyperparameter tuning
    if not tune_hyperparameters:
        # Fitting train features and labels to the pipeline
        pipe.fit(X = train_features, y = train_labels)
    else:
        # Instantiating the bayesian optimization
        bayes_search_cv = skopt.BayesSearchCV(estimator = pipe, 
                                              search_spaces = hyperparameters, 
                                              n_iter = n_iterations,
                                              scoring = metrics_dict.get(metric),
                                              n_jobs = -1, 
                                              cv = kf, 
                                              verbose = verbosity,
                                              random_state = 42)
        
        # Tuning the hyperparameters
        bayes_search_cv.fit(X = train_features, y = train_labels)
        
        # Extracting the pipeline with the best hyperparameters
        pipe = bayes_search_cv.best_estimator_
        
    # Returning the pipeline
    return pipe

# Defining a function to visually compare the predictions to ground truth labels
def visualize_predictions(model = None,
                          test_features = None,
                          test_labels = None,
                          algorithm_name = None):
    """
    This is a function to visualize the predictions of a regressor pipeline given the test features.
    
    Args:
        model: A regressor pipeline.
        test_features: Test features as a pandas data frame.
        test_labels: Ground truth labels of a test set.
        algorithm_name: A name of the algorithm used to build the regressor pipeline.
        
    Returns:
        A 2D plot.
    """
    # Making predictions given the test features
    predictions = model.predict(X = test_features)
    
    # Storing the predictions & ground truth labels in a dictionary
    dictionary = {"Ground Truth": test_labels, 
                  "Predictions": predictions}
    
    # Creating a data frame to store the dictionary and randomly sampling 10 observations
    temp_df = pd.DataFrame(data = dictionary).sample(n = 10, random_state = 42)
    
    # Visualizing the predictions of a regressor pipeline given the test features
    temp_df.plot(kind = "bar", color = ["teal", "red"], figsize = (20, 8))
    plt.title(label = f"{algorithm_name} Model Predictions vs Ground Truth", fontsize = 16)
    plt.xlabel(xlabel = "Observation Index", fontsize = 16)
    plt.ylabel(ylabel = "Price in AZN", fontsize = 16)
    plt.legend(bbox_to_anchor = (1.16, 1.019))
    plt.xticks(rotation = 0)
    plt.show()

# Defining a function to evaluate the performance of a regressor pipeline
def evaluate_performance(model = None,
                         train_features = None,
                         train_labels = None,
                         test_features = None,
                         test_labels = None,
                         metric = evaluation_metric,
                         algorithm_name = None):
    """
    This is a function to evaluate the performance of the regressor pipeline.
    
    Args:
        model: A regressor pipeline.
        train_features: Training features as a pandas data frame.
        train_labels: Ground truth labels of a train set.
        test_features: Test features as a pandas data frame.
        test_labels: Ground truth labels of a test set.
        metric: An evaluation to metric to perform the cross validation on.
        algorithm_name: A name of the algorithm used to build the regressor pipeline.
        
    Returns:
        Pandas data frame.
    """
    # Creating a dictionary of feature scalers
    metrics_dict = {"mape": "neg_mean_absolute_percentage_error",
                    "rmse": "neg_root_mean_squared_error",
                    "mae": "neg_mean_absolute_error",
                    "mse": "neg_mean_squared_error"}
    
    # Instantiating the cross validation technique
    kf = KFold(shuffle = True, random_state = 42)
    
    # Making predictions given train and test features
    train_predictions = model.predict(X = train_features)
    test_predictions = model.predict(X = test_features)
    
    # Calculating the R Squared for train & test sets
    train_r2 = model.score(X = train_features, y = train_labels)
    test_r2 = model.score(X = test_features, y = test_labels)
    
    # Defining the number of observations and features for train & test sets
    train_N, train_p = train_features.shape
    test_N, test_p = test_features.shape
    
    # Calculating the Adjusted R Squared for train & test sets
    train_adj_r2 = 1 - (((1 - train_r2) * (train_N - 1)) / (train_N - train_p - 1))
    test_adj_r2 = 1 - (((1 - test_r2) * (test_N - 1)) / (test_N - test_p - 1))
    
    # Calculating the Mean Absolute Percentage Error (MAPE) for train & test sets
    train_mape = sum(abs((train_labels - train_predictions) / train_labels)) / train_labels.size
    test_mape = sum(abs((test_labels - test_predictions) / test_labels)) / test_labels.size
    
    # Calculating the Mean Absolute Error (MAE) for train & test sets
    train_mae = sum(abs(train_labels - train_predictions)) / train_labels.size
    test_mae = sum(abs(test_labels - test_predictions)) / test_labels.size
    
    # Calculating the Mean Squared Error (MSE) for train & test sets
    train_mse = sum(pow(base = train_labels - train_predictions, exp = 2)) / train_labels.size
    test_mse = sum(pow(base = test_labels - test_predictions, exp = 2)) / test_labels.size
    
    # Calculating the Root Mean Squared Error (RMSE) for train & test sets
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    # Calculating the cross validation scores
    cv_scores = abs(cross_val_score(estimator = model, 
                                    X = train_features, 
                                    y = train_labels, 
                                    n_jobs = -1))
    
    # Calculating the maximum cross validation score
    max_cv_score = cv_scores[np.argmax(a = cv_scores)]
    
    # Calculating the minimum cross validation score
    min_cv_score = cv_scores[np.argmin(a = cv_scores)]
    
    # Calculating the average cross validation score
    mean_cv_score = cv_scores.mean()
    
    # Calculating the standard deviation of cross validation scores
    std_cv_score = cv_scores.std()
    
    # Creating a dictionary to store evaluate metrics
    evaluation_metrics_dict = {"Train R2": train_r2,
                               "Test R2": test_r2,
                               "Train Adjusted R2": train_adj_r2,
                               "Test Adjusted R2": test_adj_r2,
                               "Train MAE": train_mae,
                               "Test MAE": test_mae,
                               "Train MAPE": train_mape,
                               "Test MAPE": test_mape,
                               "Train MSE": train_mse,
                               "Test MSE": test_mse,
                               "Train RMSE": train_rmse,
                               "Test RMSE": test_rmse,
                               "Mean CV": mean_cv_score,
                               "Max CV": max_cv_score,
                               "Min CV": min_cv_score,
                               "Std CV": std_cv_score}
    
    # Storing the dictionary in a data frame
    evaluation_df = pd.DataFrame(data = evaluation_metrics_dict, index = [algorithm_name])
    
    # Rounding the precision of the metrics
    evaluation_df = evaluation_df.applymap(func = lambda x: round(number = x, ndigits = 2))
    
    # Returning the data frame
    return evaluation_df