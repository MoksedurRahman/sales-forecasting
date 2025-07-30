# %% [markdown]
# 1. Importing Libraries

# %%
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose  

from sklearn.model_selection import train_test_split

# %% [markdown]
# 2. Loading Dataset

# %%
# Load data
sales = pd.read_csv('../data/stores_sales_forecasting.csv', encoding='latin1')


# %% [markdown]
# 3. Data Exploration

# %%
sales.head(5)

# %%
sales.tail()

# %%
sales.shape

# %%
sales.info()

# %%
sales.describe()

# %%
sales.isnull().sum()

# %%
dict = {}
for i in list(sales.columns):
    dict[i] = sales[i].value_counts().shape[0]

pd.DataFrame(dict,index=["unique count"]).transpose()

# %%
sales = sales.drop(columns=['Row ID'])
sales.head(3)

# %%
sales = sales.drop(columns=['Order ID', 'Country', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 
                         'Postal Code', 'Product ID', 'Product Name', 'Sub-Category', 'Category', 'Region', 'State'])
sales.head(3)

# %%
duplicate_rows = sales[sales.duplicated()]
duplicate_rows.head()

# %%
sales = sales.drop_duplicates()
sales.head()

# %%
sales['Order Date'] = pd.to_datetime(sales['Order Date'])
sales.set_index('Order Date', inplace=True)

# %%
monthly_sales = sales['Sales'].resample('M').sum()

# %%
monthly_sales

# %% [markdown]
# 5. Data Visualization

# %%
plt.figure(figsize=(15, 5))

# Sales Distribution
plt.subplot(1, 3, 1)
sns.histplot(sales['Sales'], bins=30, kde=True)
plt.title('Sales Distribution')

# Quantity Distribution
plt.subplot(1, 3, 2)
sns.histplot(sales['Quantity'], bins=30, kde=True)
plt.title('Quantity Distribution')

# Profit Distribution
plt.subplot(1, 3, 3)
sns.histplot(sales['Profit'], bins=30, kde=True)
plt.title('Profit Distribution')

plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', linestyle='-')
plt.title('Monthly Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# %%
scatter_plot = px.scatter(sales, x='Sales', y='Profit', color='Sales', size='Discount', title='Sales vs Profit')
scatter_plot.update_layout(
    width=800,  
    height=600  
)
scatter_plot.show()

# %%
plt.figure(figsize=(6, 4))
plt.hist(sales['Sales'], bins=20, color='skyblue', edgecolor='black')
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# %%
# Seasonal Decomposition Plot
decomposition = seasonal_decompose(monthly_sales, model='additive')
plt.figure(figsize=(7,5))

plt.subplot(4, 1, 1)
plt.plot(decomposition.trend, label='Trend', color='blue')
plt.title('Trend')

plt.subplot(4, 1, 2)
plt.plot(decomposition.seasonal, label='Seasonality', color='green')
plt.title('Seasonality')

plt.subplot(4, 1, 3)
plt.plot(decomposition.resid, label='Residuals', color='red')
plt.title('Residuals')

plt.subplot(4, 1, 4)
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', linestyle='-', label='Original', color='black')
plt.title('Original')

plt.tight_layout()
plt.show()

# %%
# Resample the data to compute annual average of the 'Sales' column
Y_Sales = sales['Sales'].resample('Y').mean()

Y_Sales.plot(figsize =(3,3))

# %%
# Resample the data to compute annual average of the 'Profit' column
Y_Profit = sales['Profit'].resample('Y').mean()

Y_Profit.plot(figsize =(3,3), c= "g")

# %%
# Resample the data to compute annual average of the 'Quantity' column
Y_Quantity = sales['Quantity'].resample('Y').mean()

Y_Quantity.plot(figsize =(3,3), c ="r")

# %%
# Resample the data to compute annual average of the 'Discount' column.
Y_Discount = sales['Discount'].resample('Y').mean()

Y_Discount.plot(figsize =(3,3), c ="pink", )

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_result = seasonal_decompose(monthly_sales, model='multiplicative', period=12)

# %% [markdown]
# 6. ADF Test

# %%
# Check for stationarity
from statsmodels.tsa.stattools import adfuller
adfuller_result = adfuller(monthly_sales)
print("ADF Statistic: %f" % adfuller_result[0])
print('p-value: %f' % adfuller_result[1])

# %%
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(sales['Sales'])
plt.show()

# %% [markdown]
# 7. ACF & PACF

# %%
import statsmodels.api as sm

# Plot ACF
fig, ax1 = plt.subplots(figsize=(12, 6))
sm.graphics.tsa.plot_acf(sales['Sales'], lags=40, ax=ax1)
plt.title('Autocorrelation Function (ACF)')

# Plot PACF
fig, ax2 = plt.subplots(figsize=(12, 6))
sm.graphics.tsa.plot_pacf(sales['Sales'], lags=40, ax=ax2)
plt.title('Partial Autocorrelation Function (PACF)')

plt.show()

# %% [markdown]
# 8. Model Building

# %%
# Split data into train and test sets
X = np.arange(len(monthly_sales)).reshape(-1, 1)
y = monthly_sales.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# %%
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit ARIMA model
arima_model = ARIMA(y_train, order=(5,1,0))
arima_model_fit = arima_model.fit()

# Fit SARIMA model
sarima_model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_model_fit = sarima_model.fit()

# %%
arima_model_fit.summary()

# %%
sarima_model_fit.summary()

# %% [markdown]
# 9. Forecasting

# %%
arima_predictions = arima_model_fit.forecast(steps=len(X_test))
sarima_predictions = sarima_model_fit.forecast(steps=len(X_test))
# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, label='Train')
plt.plot(X_test, y_test, label='Test')
plt.plot(X_test, arima_predictions, label='ARIMA Predictions')
plt.plot(X_test, sarima_predictions, label='SARIMA Predictions')
plt.title('Sales Forecasting with ARIMA and SARIMA')
plt.xlabel('Months')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


