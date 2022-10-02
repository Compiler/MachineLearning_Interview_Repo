import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import  train_test_split
import os, pathlib
data_base = 'data/predict_housing_price_data'
train_path = f'{data_base}/train.csv'
test_path = f'{data_base}/test.csv'
data_path = f'{data_base}/california_housing_train.csv'

assert(os.path.exists(data_path))
df = pd.read_csv(data_path)
df['median_income_to_house_value'] = df['median_house_value'] / df['median_income']
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

train, test = train_test_split(df, test_size = 0.2)


df.info()

np.sum(df.isnull())

df.describe()

corr_matrix = train.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)


#sns.pairplot(df)

train_copy = train
before_shape = train_copy.shape
print(train_copy.shape)
upper_bounds = {}
lower_bounds = {}
for key in train_copy:
    q25, q50, q75, q90 = np.percentile(train_copy[key], [25, 50, 75, 90])
    inter_quartile_range = q75 - q25
    upper = q75 + 1.5*inter_quartile_range
    lower = q25 - 1.5*inter_quartile_range
    upper_bounds[key] = upper
    lower_bounds[key] = lower
    train_copy = train_copy[(train_copy[key] >= lower) & (train_copy[key] <= upper)].reset_index(drop=True)

print(before_shape[0] - train_copy.shape[0], 'removed outliers')
    

train = train_copy.copy()

features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'rooms_per_household', 'bedrooms_per_room', 'population_per_household']
print(features)

target = ['median_house_value']

train_X, train_Y = train[features], train[target]

def vif(X, vif_lim = 10):
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    for i in range(len(X.columns)):
        
        l = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        s = pd.Series(index=X.columns, data=l).sort_values(ascending=False)
        
        if( s.iloc[0] > vif_lim):
            X.drop(s.index[0], axis=1, inplace=True)
            print("Removed",s.index[0],', VIF = ', s.iloc[0])
        else:
            break
    print("Remaining columns:", X.columns)

x_copy = train_X.copy()

vif(x_copy)
train_X = x_copy.copy()


#fit model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


folds = 5
lr = LinearRegression()
scores = cross_val_score(lr, train_X, train_Y, cv=folds, scoring='r2')
print(f'r2={np.sqrt(scores).mean()}')


lr = LinearRegression()
scores = cross_val_score(lr, train_X, train_Y, cv=folds, scoring='neg_mean_squared_error')
print(f'neg_mean_squared_error={np.sqrt(-scores).mean()}')

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.1)

scores = cross_val_score(ridge, train_X, train_Y, cv=5, scoring='r2')
print(f'r2={np.sqrt(scores).mean()}')

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)

scores = cross_val_score(lasso, train_X, train_Y, cv=5, scoring='r2')
print(f'r2={np.sqrt(scores).mean()}')

from sklearn.model_selection import validation_curve

ridge = Ridge(random_state=42)
param_name = 'alpha'
param_range=[0, 0.1, 1, 10, 50, 100, 200, 300, 500, 700]
scoring = 'neg_mean_squared_error'

scores = validation_curve(ridge, train_X, train_Y, scoring=scoring,cv=5,param_name=param_name, param_range=param_range)

train_score=[np.sqrt(-scores[0][i].mean()) for i in range(len(param_range))]
test_score=[np.sqrt(-scores[1][i].mean()) for i in range(len(param_range))]




fig = plt.figure(figsize=(8,6))
plt.plot(param_range, train_score, label='Train')
plt.plot(param_range, test_score, label='Test')
plt.xticks=param_range
plt.title(f"Validation curve of {param_name}")
plt.legend()

lr = LinearRegression()
lr.fit(train_X, train_Y)

l = list(train_X.columns)
l.extend(target)

features_used = list(train_X.columns)
features_used.extend(target)

test_copy = test.copy()
test_copy = test_copy[features_used]

for c in test_copy.columns:
    print(c)
    test_copy = test_copy[(test_copy[c] >= lower) & (test_copy[c] <= upper)].reset_index(drop=True)

test_X, test_Y = test_copy[train_X.columns], test_copy[target]
print(train_X.columns)

train_mse = mean_squared_error(train_Y, lr.predict(train_X))
train_rmse = np.sqrt(train_mse)
print("Training error:", train_rmse)

print(test_copy.info())
y_pred = lr.predict(test_X)
test_mse = mean_squared_error(y_pred, test_Y)
test_rmse = np.sqrt(test_mse)
print("Testing error:", test_rmse)

