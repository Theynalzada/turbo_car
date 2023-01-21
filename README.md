# Data Science Project: Car Price Prediction

## Project Description
The main goal of this project is to build a regression model that is capable of predicting the approximate price of a car given input variables. 

## Data Collection
Data has been collected from **https://www.turbo.az** website using **requests** and **BeautifulSoup** libraries in **Python** programming language. Initial dataset contains **14044** unique observations from **9** types of vehicles with **87** brands and **822** models which you can find the detailed information in the following table that has been aggregated based on the vehicle type.

| Vehicle Type | Number of Observations | Number of Brands | Number of Models |
| -------- | ---------------------- | ---------------- | ---------------- |
| Hatchback|    1234                |	40			   |	131		      |
| Cabriolet|      11                |	4			   |	9		      |
| Coupe    |     131                |	21			   |	52		      |
| Liftback |      386               |	17			   |	43		      |
| Offroader / SUV|   4362           |	63			   |	320		      |
| Pickup   |       127              |	19			   |	23		      |
| Roadster |        9               |	5			   |	8		      |
| Sedan    |       6927             |	61			   |	349		      |
| Universal|      857               |	20			   |	59		      |

The variables used for modeling are following.

- **is_new** - A binary variable indicating whether or not a car is new
- **is_luxury_brand** - A binary variable indicating whether or not a brand a car belongs to is luxury
- **exchange_available** - A binary variable indicating the availablity of an exchange
- **is_saloon_car** - A binary variable indicating whether or not a car is being sold by a particular saloon
- **loan_available** - A binary variable indicating the availablity of a loan
- **is_overdriven** - A binary variable indicating whether or not a mileage is greater equal than 150000 km
- **has_alloy_wheels** - A binary variable indicating whether or not a car has alloy wheels
- **has_abs** - A binary variable indicating whether or not a car has anti braking system
- **has_hatch** - A binary variable indicating whether or not a car has hatch
- **has_rain_sensor** - A binary variable indicating whether or not a car has rain sensor
- **has_central_locking** - A binary variable indicating whether or not a car has central locking
- **has_parking_radar** - A binary variable indicating whether or not a car has parking radar
- **has_air_conditioner** - A binary variable indicating whether or not a car has air conditioner
- **has_seat_heating** - A binary variable indicating whether or not a car has seat heating
- **has_leather_salon** - A binary variable indicating whether or not a car has leather salon
- **has_xenon_lamps** - A binary variable indicating whether or not a car has xenon lamps
- **has_rear_view_camera** - A binary variable indicating whether or not a car has rear view camera
- **has_side_curtains** - A binary variable indicating whether or not a car has side curtains
- **has_seat_ventilation** - A binary variable indicating whether or not a car has seat ventilation
- **vehicle_type** - The type of a vehicle
- **brand** - The name of a brand
- **origin** - The origin of a brand
- **color** - The color of a vehicle
- **fuel_type** - The fuel type of a vehicle
- **speed_box** - The type of a speed box
- **transmission** - The type of a transmission
- **age** - The age of a vehicle
- **engine** - The engine of a vehicle
- **mileage** - The mileage of a vehicle
- **hp** - The horse power of a vehicle
- **avg_mileage_per_year** - The average mileage per year

## Modeling
Since it is a regression problem the main evalution metric is **Mean Squared Error (MSE)** and the dependent variable has been converted to the local currency. The independent features used in modeling are categorized into four different categories.

1. **Nominal Features**
2. **High Cardinality Features**
3. **Binary Features**
4. **Numeric Features**

For the curse of dimensionality problem it would not be a good choice to use a nominal variable with too many unique categories because after creating a one hot array there would be too many variables. In order to handle this problem, I created a custom **FrequencyRatioEncoder** transformer. In order to handle the low frequency categories in nominal variables I created another custom **RareLabelEncoder** transformer. Since multicollinearity among numeric features is something that needs to be taken care of when using a linear algorithm, I created a custom **VIFDropper** transformer that removes highly correlated features based on high **variance inflation factor (VIF)** value. In addition, I created **PhiCoefficientDropper** transformer to drop multicollinear binary features with with a threshold of **60%** (inclusive) for linear algorithms. The hyper parameters have been tuned using **Bayesian Optimization** technique while the log transformation has been applied to the dependent variable for some algorithms. Since **Light Gradient Boosting (LightGBM), Extreme Gradient Boosting Machine (XGBoost)** and **Category Boosting (CatBoost)** have outperformed the models built with other algorithms, I used these models to build **stacked** and **voting** model while the latter had the lowest **MSE** on the test set.

| Algorithm | Selected | Log Transformation Applied |
| --------                 | -------- | -------- |
| Baseline                 | False    | False    |
| Linear Regression        | False    | True     |
| Bayesian Ridge           | False    | True     |
| Support Vector Machine   | False    | True     |
| K Nearest Neighbors      | False    | True     |
| Multi Layer Perceptron   | False    | True     |
| Decision Tree            | False    | False    |
| Random Forest            | False    | False    |
| Adaptive Boosting        | False    | False    |
| Gradient Boosting        | False    | False    |
| Light Gradient Boosting  | True     | False    |
| Extreme Gradient Boosting| True     | False    |
| Category Boosting        | True     | False    |


The image below compares the predicted prices by **Light Gradient Boosting Machine (LightGBM)** algorithm to the actual price of randomly selected 10 cars from the test.

![](https://i.imgur.com/yiUThyM.png)

The image below compares the predicted prices by **Extreme Gradient Boosting Machine (XGBoost)** algorithm to the actual price of randomly selected 10 cars from the test.

![](https://i.imgur.com/Q58pEx9.png)

The image below compares the predicted prices by **Category Boosting (CatBoost)** algorithm to the actual price of randomly selected 10 cars from the test.

![](https://i.imgur.com/TYlQ0jv.png)

The image below compares the predicted prices by **Voting** algorithm to the actual price of randomly selected 10 cars from the test.

![](https://i.imgur.com/QzkVHGB.png)

The image below is the visualization of the pipeline for the **LightGBM** model.

![](https://i.imgur.com/uWsF4KH.png)

The image below is the visualization of the pipeline for the **XGBoost** model.

![](https://i.imgur.com/1hRW9MY.png)

The image below is the visualization of the pipeline for the **CatBoost** model.

![](https://i.imgur.com/FBtpIzb.png)

All these pipelines together have been used to build a final regression pipeline using **Voting** algorithm which prevailed models built with other algorithms. The evaluation metrics for the **LightGBM, XGBoost, CatBoost** and **Voting** models are in the following table.

| Metrics          | LightGBM | XGBoost | CatBoost | Voting |
| ---------------- | -------- | ------- | -------- | ------ |
| Train R2         | 0.98     | 0.99    | 0.98     | 0.99   |
| Test R2          | 0.94  	  |	0.93    | 0.94	   | 0.94   |
| Train Adjusted R2| 0.98  	  |	0.99    | 0.98	   | 0.99   |
| Test Adjusted R2 | 0.94  	  |	0.93    | 0.94	   | 0.94   |
| Train MAE        | 3991.64  |	2771.06 | 4276.42  | 3444.92|
| Test MAE         | 5079.78  |	4624.82 | 5172.31  | 4661.81|
| Train MAPE       | 0.14 	  |	0.11	| 0.15	   | 0.13   |
| Test MAPE        | 0.17	  |	0.15	| 0.18	   | 0.16   |
| Train MSE        | 58278601.6|2.216382e+07| 6.386657e+07| 3.917994e+07|
| Test MSE         | 1.610828e+08|1.880002e+08| 1.786054e+08| 1.601509e+08|
| Train RMSE       | 7634.04  |	4707.85 | 7991.66  | 6259.39|
| Test RMSE        | 12691.84 |	13711.32| 13364.33 | 12655.07|
| Mean CV          | 0.93  	  |	0.94	| 0.94	   | 0.94   |
| Max CV           | 0.95     |	0.95	| 0.95	   | 0.95   |
| Min CV           | 0.92  	  |	0.93	| 0.92	   | 0.93   |
| Std CV           | 0.01  	  |	0.01	| 0.01	   | 0.01   |