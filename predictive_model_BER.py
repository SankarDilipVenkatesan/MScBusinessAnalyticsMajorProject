# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read the BER rating dataset
df = pd.read_csv("sample_dataset_BER_rating.csv")

# Convert the "Year_of_Construction" column to datetime format
df['Year_of_Construction'] = pd.to_datetime(df['Year_of_Construction'], format='%Y')

# Calculate the average BER rating for each year
df_yearly_avg = df.groupby(df['Year_of_Construction'].dt.year)['BerRating'].mean()

# Plot the average BER ratings over time
plt.figure(figsize=(10, 6))
plt.plot(df_yearly_avg)
plt.title('Average Building Energy Rating (BER) over Time')
plt.xlabel('Year')
plt.ylabel('Average BER Rating')
plt.show()

# Perform differencing and plot the differenced series
diff = df_yearly_avg.diff().dropna()
plt.figure(figsize=(10, 4))
plt.plot(diff)
plt.title('Differenced Average BER Rating')
plt.xlabel('Year')
plt.ylabel('Difference')
plt.show()

# Calculate autocorrelation and partial autocorrelation plots to determine SARIMA parameters
plot_acf(diff)
plt.title('Average BER Rating Autocorrelation')
plt.show()

plot_pacf(diff)
plt.title('Average BER Rating Partial Autocorrelation')
plt.show()

# Define the SARIMA parameters based on the plots and domain knowledge
order = (2, 1, 0)             
seasonal_order = (2, 0, 2, 12) 

# Fit the SARIMA model
model = SARIMAX(df_yearly_avg, order=order, seasonal_order=seasonal_order)
result = model.fit()

# Predict future values
future_years = pd.date_range(start='1753', end='2030', freq='YS')
predictions = result.get_forecast(steps=len(future_years))
predicted_values = predictions.predicted_mean

# Plot the predicted average BER ratings for 2030
plt.figure(figsize=(10, 6))
plt.plot(df_yearly_avg, label='Historical')
plt.plot(predicted_values, label='Predicted')
plt.title('Average Building Energy Rating (BER) Prediction for 2030')
plt.xlabel('Year')
plt.ylabel('Average BER Rating')
plt.legend()
plt.show()

# %%
print(predicted_values)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read the BER rating dataset
df = pd.read_csv("sample_dataset_BER_rating.csv")

# Convert the "Year_of_Construction" column to datetime format
df['Year_of_Construction'] = pd.to_datetime(df['Year_of_Construction'], format='%Y')

# Calculate the average BER rating for each year
df_yearly_avg = df.groupby(df['Year_of_Construction'].dt.year)['BerRating'].mean()

# Plot the average BER ratings over time
plt.figure(figsize=(10, 6))
plt.plot(df_yearly_avg)
plt.title('Average Building Energy Rating (BER) over Time')
plt.xlabel('Year')
plt.ylabel('Average BER Rating')
plt.show()

# Perform differencing and plot the differenced series
diff = df_yearly_avg.diff().dropna()
plt.figure(figsize=(10, 4))
plt.plot(diff)
plt.title('Differenced Average BER Rating')
plt.xlabel('Year')
plt.ylabel('Difference')
plt.show()

# Calculate autocorrelation and partial autocorrelation plots to determine SARIMA parameters
plot_acf(diff)
plt.title('Average BER Rating Autocorrelation')
plt.show()

plot_pacf(diff)
plt.title('Average BER Rating Partial Autocorrelation')
plt.show()

# Define the SARIMA parameters based on the plots and domain knowledge
order = (2, 1, 0)             
seasonal_order = (2, 0, 2, 12) 

# Fit the SARIMA model
model = SARIMAX(df_yearly_avg, order=order, seasonal_order=seasonal_order)
result = model.fit()

# Predict future values
future_years = pd.date_range(start='2025-01-01', end= '2031-01-01', freq='AS')
predictions = result.get_forecast(steps=len(future_years))
predicted_values = predictions.predicted_mean

# Plot the predicted average BER ratings for 2030
plt.figure(figsize=(10, 6))
plt.plot(df_yearly_avg.index, df_yearly_avg, label='Historical')
plt.plot(predicted_values.index, predicted_values, label='Predicted')
plt.title('Average Building Energy Rating (BER) Prediction for 2030')
plt.xlabel('Year')
plt.ylabel('Average BER Rating')
plt.legend()
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read the BER rating dataset
df = pd.read_csv("sample_dataset_BER_rating.csv")

# Convert the "Year_of_Construction" column to datetime format
df['Year_of_Construction'] = pd.to_datetime(df['Year_of_Construction'], format='%Y')

# Calculate the average BER rating for each year
df_yearly_avg = df.groupby(df['Year_of_Construction'].dt.year)['BerRating'].mean()

# Plot the average BER ratings over time
plt.figure(figsize=(10, 6))
plt.plot(df_yearly_avg)
plt.title('Average Building Energy Rating (BER) over Time')
plt.xlabel('Year')
plt.ylabel('Average BER Rating')
plt.show()

# Perform differencing and plot the differenced series
diff = df_yearly_avg.diff().dropna()
plt.figure(figsize=(10, 4))
plt.plot(diff)
plt.title('Differenced Average BER Rating')
plt.xlabel('Year')
plt.ylabel('Difference')
plt.show()

# Calculate autocorrelation and partial autocorrelation plots to determine SARIMA parameters
plot_acf(diff)
plt.title('Average BER Rating Autocorrelation')
plt.show()

plot_pacf(diff)
plt.title('Average BER Rating Partial Autocorrelation')
plt.show()

# Define the SARIMA parameters based on the plots and domain knowledge
order = (2, 1, 0)             
seasonal_order = (2, 0, 2, 12) 

# Fit the SARIMA model
model = SARIMAX(df_yearly_avg, order=order, seasonal_order=seasonal_order)
result = model.fit()

# Predict future values
future_years = [2025, 2026, 2027, 2028, 2029, 2030]
predicted_values = result.get_forecast(steps=len(future_years)).predicted_mean
predicted_values.index = future_years

# Plot the predicted average BER ratings for 2030
plt.figure(figsize=(10, 6))
plt.plot(df_yearly_avg.index, df_yearly_avg, label='Historical')
plt.plot(predicted_values.index, predicted_values, label='Predicted')
plt.title('Average Building Energy Rating (BER) Prediction for 2030')
plt.xlabel('Year')
plt.ylabel('Average BER Rating')
plt.legend()
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read the BER rating dataset
df = pd.read_csv("sample_dataset_BER_rating.csv")

# Convert the "Year_of_Construction" column to datetime format
df['Year_of_Construction'] = pd.to_datetime(df['Year_of_Construction'], format='%Y')

# Calculate the average BER rating for each year
df_yearly_avg = df.groupby(df['Year_of_Construction'].dt.year)['BerRating'].mean()

# Plot the average BER ratings over time
plt.figure(figsize=(10, 6))
plt.plot(df_yearly_avg)
plt.title('Average Building Energy Rating (BER) over Time')
plt.xlabel('Year')
plt.ylabel('Average BER Rating')
plt.show()

# Perform differencing and plot the differenced series
diff = df_yearly_avg.diff().dropna()
plt.figure(figsize=(10, 4))
plt.plot(diff)
plt.title('Differenced Average BER Rating')
plt.xlabel('Year')
plt.ylabel('Difference')
plt.show()

# Calculate autocorrelation and partial autocorrelation plots to determine SARIMA parameters
plot_acf(diff)
plt.title('Average BER Rating Autocorrelation')
plt.show()

plot_pacf(diff)
plt.title('Average BER Rating Partial Autocorrelation')
plt.show()

# Define the SARIMA parameters based on the plots and domain knowledge
order = (2, 1, 0)             # Replace p, d, q with the appropriate values
seasonal_order = (2, 0, 2, 12) # Replace P, D, Q, s with the appropriate values

# Fit the SARIMA model
model = SARIMAX(df_yearly_avg[:2025], order=order, seasonal_order=seasonal_order)
result = model.fit()

# Predict future values
future_years = [2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
predicted_values = result.get_forecast(steps=len(future_years)).predicted_mean
predicted_values.index = future_years

# Plot the predicted average BER ratings from 1990 to 2024 and the predicted values from 2025 to 2030
plt.figure(figsize=(10, 6))
plt.plot(df_yearly_avg.loc[1990:], label='Historical (1990-2022)')
plt.plot(predicted_values, label='Predicted (2022-2030)')
plt.title('Average Building Energy Rating (BER) Prediction')
plt.xlabel('Year')
plt.ylabel('Average BER Rating')
plt.legend()

# Add a label for the predicted average BER rating in 2030 with a customized arrow location
predicted_2030 = predicted_values.loc[2030]
arrow_x = 2029.5  # X-coordinate for the arrow location
arrow_y = predicted_2030 + 6  # Y-coordinate for the arrow location
text_x = 2024  # X-coordinate for the text label location
text_y = predicted_2030 + 25  # Y-coordinate for the text label location

plt.annotate(f'2030: {predicted_2030:.2f}', xy=(2030, predicted_2030), xytext=(text_x, text_y),
             arrowprops=dict(arrowstyle='->'), xycoords='data', textcoords='data')


plt.show()


