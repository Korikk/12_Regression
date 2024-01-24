import pandas as pd
import numpy as np
from scipy.stats import shapiro

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("Datasets/Advertising.csv")
df = df_.copy()
df.head(10)
df.info()
df.describe().T

# Creating formulas with pre-defined weights.
# y = b + w1*x1 + w2*x2 + w3*x3
# y1 = 2.90 + 0.04*TV + 0.17*radio + 0.002*newspaper
# y2 = 1.70 + 0.09*TV + 0.20*radio + 0.017*newspaper


# A function that calculates target value for each row in df. Returns output column.
def calculate_y(dataframe, features, weights):
    return weights[0] + dataframe[features[0]]*weights[1] + dataframe[features[1]]*weights[2] + \
           dataframe[features[2]]*weights[3]


df.columns
variables = ['TV', 'radio', 'newspaper']
weights1 = [2.90, 0.04, 0.17, 0.002]
weights2 = [1.70, 0.09, 0.20, 0.017]

df['y_hat_1'] = calculate_y(df, variables, weights1)
df['y_hat_2'] = calculate_y(df, variables, weights2)

# Calculate differences between true and predicted target values.
df['diff_1'] = df['sales'] - df['y_hat_1']
df['diff_2'] = df['sales'] - df['y_hat_2']

df.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T


# a function that calculates mean squared error.
def mse(dataframe, diff_column):
    return dataframe[diff_column].pow(2).sum() / dataframe.shape[0]


mse_1 = mse(df, 'diff_1')
mse_2 = mse(df, 'diff_2')


# a function that calculates root mean square error
def rmse(dataframe, diff_column):
    return np.sqrt(mse(dataframe, diff_column))


rmse_1 = rmse(df, 'diff_1')
rmse_2 = rmse(df, 'diff_2')


# a function that calculates mean absolute error
def mae(dataframe, diff_column):
    return dataframe[diff_column].abs().sum() / dataframe.shape[0]


mae_1 = mae(df, 'diff_1')
mae_2 = mae(df, 'diff_2')

df_errors = pd.DataFrame([mse_1, mse_2, rmse_1, rmse_2, mae_1, mae_2],
                         index=['mse_1', 'mse_2', 'rmse_1', 'rmse_2', 'mae_1', 'mae_2'])
test_statistic_1, p_value_1 = shapiro(df['y_hat_1'])
print('Test Statistic = %.4f, p-value = %.4f' % (test_statistic_1, p_value_1))
p_value_1 < 0.05
test_statistic_2, p_value_2 = shapiro(df['y_hat_2'])
print('Test Statistic = %.4f, p-value = %.4f' % (test_statistic_2, p_value_2))
p_value_2 < 0.05
# Both p values are lower than 0.05 so we reject H0, assumption of Normality is not provided.
# Although x is rejected, the mean value can be used as there is little difference between the median and the mean value

percentage1 = (rmse_1 / df['y_hat_1'].mean()) * 100
percentage2 = (rmse_2 / df['y_hat_2'].mean()) * 100

# These percentages can be give an idea about magnitudes of errors.
