# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %%
df = pd.read_csv("production_monthly_data.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%Y')
df.set_index('Date', inplace=True)
print(df)

# %%
plt.figure(figsize=(10, 6))
plt.plot(df['Renewables'])
plt.title('Electricity Generation in Ireland')
plt.xlabel('Date')
plt.ylabel('Renewables')
plt.show()

# %%
acf_plot = plot_acf(df["Renewables"], lags=100)

# %%
# Perform differencing and plot the differenced series
diff = df['Renewables'].diff().dropna()
plt.figure(figsize=(10, 4))
plt.plot(diff)
plt.title('Differenced Non-Renewable Electricity Generation')
plt.xlabel('Date')
plt.ylabel('Difference')
plt.show()

# Calculate autocorrelation and partial autocorrelation plots to determine SARIMA parameters
plot_acf(diff)
plt.title('Renewables Autocorrelation')
plt.show()

plot_pacf(diff)
plt.title('Renewables Partial Autocorrelation')
plt.show()

# %% [markdown]
# PACF = 1, 5, 6, 7, 8, 9

# %%
# Define the SARIMA parameters based on the plots and your domain knowledge
order = (2, 1, 0)            # Replace p, d, q with the appropriate values
seasonal_order = (2, 0, 2, 12)  # Replace P, D, Q, s with the appropriate values

# Fit the SARIMA model
model = SARIMAX(df['Renewables'], order=order, seasonal_order=seasonal_order)
result = model.fit()

# Predict future values
future_dates = pd.date_range(start='2022-11-01', end='2030-12-01', freq='MS')  # Replace the date range as needed
predictions = result.get_forecast(steps=len(future_dates))
predicted_values = predictions.predicted_mean

# %%
plt.figure(figsize=(10, 6))
plt.plot(df['Renewables'], label='Historical')
plt.plot(predicted_values, label='Predicted')
plt.title('Renewable Electricity Generation Prediction')
plt.xlabel('Date')
plt.ylabel('Renewables (ktoe)')
plt.legend()
plt.show()

# %%
# Define the SARIMA parameters based on the plots and your domain knowledge
order = (2, 1, 0)            # Replace p, d, q with the appropriate values
seasonal_order = (2, 0, 0, 12)  # Replace P, D, Q, s with the appropriate values

# Fit the SARIMA model
model = SARIMAX(df['Renewables'], order=order, seasonal_order=seasonal_order)
result = model.fit()

# Predict future values
future_dates = pd.date_range(start='2022-11-01', end='2030-12-01', freq='MS')  # Replace the date range as needed
predictions = result.get_forecast(steps=len(future_dates))
predicted_values = predictions.predicted_mean

# %%
plt.figure(figsize=(10, 6))
plt.plot(df['Renewables'], label='Historical')
plt.plot(predicted_values, label='Predicted')
plt.title('Non-Renewable Electricity Generation Prediction')
plt.xlabel('Date')
plt.ylabel('Renewables')
plt.legend()
plt.show()


