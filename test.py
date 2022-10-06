import pandas as pd
from sklearn.model_selection import  train_test_split

data_base = 'data/predict_housing_price_data'
data_path = f'{data_base}/california_housing_train.csv'
df = pd.read_csv(data_path)
print('Iterated!')