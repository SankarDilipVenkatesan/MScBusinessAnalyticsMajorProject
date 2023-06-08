# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns

# %%
# Read the data and set the index
df = pd.read_csv("production_monthly_data.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%Y')
df.set_index('Date', inplace=True)

# %%
# Plot Non-Renewables electricity generation
plt.figure(figsize=(10, 6))
plt.plot(df['Non-Renewables'], label='Non-Renewables')
plt.plot(df['Renewables'], label='Renewables')
plt.title('Electricity Generation in Ireland')
plt.xlabel('Date')
plt.ylabel('Electricity Generation (ktoe)')
plt.legend()
plt.show()

# %%
# Perform differencing and plot the differenced series
diff_non_renewables = df['Non-Renewables'].diff().dropna()
diff_renewables = df['Renewables'].diff().dropna()

# %%
plt.figure(figsize=(10, 4))
plt.plot(diff_non_renewables, label='Non-Renewables')
plt.plot(diff_renewables, label='Renewables')
plt.title('Differenced Electricity Generation')
plt.xlabel('Date')
plt.ylabel('Difference')
plt.legend()
plt.show()

# %%
# Calculate autocorrelation and partial autocorrelation plots for Non-Renewables
plot_acf(diff_non_renewables)
plt.title('Non-Renewables Autocorrelation')
plt.show()

plot_pacf(diff_non_renewables)
plt.title('Non-Renewables Partial Autocorrelation')
plt.show()

# Calculate autocorrelation and partial autocorrelation plots for Renewables
plot_acf(diff_renewables)
plt.title('Renewables Autocorrelation')
plt.show()

plot_pacf(diff_renewables)
plt.title('Renewables Partial Autocorrelation')
plt.show()

# %%
# Define the SARIMA parameters based on the plots and your domain knowledge
order_non_renewables = (2, 1, 0)            # Replace p, d, q with the appropriate values for Non-Renewables
order_renewables = (2, 1 ,0 )            # Replace p, d, q with the appropriate values for Renewables
seasonal_order_non_renewables = (2, 0, 12, 12)  # Replace P, D, Q, s with the appropriate values for Non-Renewables
seasonal_order_renewables = (2, 0, 2, 12)  # Replace P, D, Q, s with the appropriate values for Renewables

# %%
# Fit the SARIMA model for Non-Renewables
model_non_renewables = SARIMAX(df['Non-Renewables'], order=order_non_renewables, seasonal_order=seasonal_order_non_renewables)
result_non_renewables = model_non_renewables.fit()

# Fit the SARIMA model for Renewables
model_renewables = SARIMAX(df['Renewables'], order=order_renewables, seasonal_order=seasonal_order_renewables)
result_renewables = model_renewables.fit()

# %%
# Predict future values for Non-Renewables
future_dates = pd.date_range(start='2022-11-01', end='2030-12-01', freq='MS')  # Replace the date range as needed
predictions_non_renewables = result_non_renewables.get_forecast(steps=len(future_dates))
predicted_values_non_renewables = predictions_non_renewables.predicted_mean

# Predict future values for Renewables
predictions_renewables = result_renewables.get_forecast(steps=len(future_dates))
predicted_values_renewables = predictions_renewables.predicted_mean

# %%
# Plot the predicted values
plt.figure(figsize=(10, 6))
plt.plot(df['Non-Renewables'], label='Non-Renewables (Historical)')
plt.plot(df['Renewables'], label='Renewables (Historical)')
plt.plot(future_dates, predicted_values_non_renewables, label='Non-Renewables (Predicted)')
plt.plot(future_dates, predicted_values_renewables, label='Renewables (Predicted)')
plt.title('Electricity Generation Prediction')
plt.xlabel('Date')
plt.ylabel('Electricity Generation (ktoe)')
plt.legend()
plt.show()

# %%
# Calculate the average amount at the year 2030
average_non_renewables_2030 = np.mean(predicted_values_non_renewables)
average_renewables_2030 = np.mean(predicted_values_renewables)

# %%
print("Average Non-Renewables in 2030:", average_non_renewables_2030)
print("Average Renewables in 2030:", average_renewables_2030)

# %%
# Create a pie chart for the averages
averages_2030 = [average_non_renewables_2030, average_renewables_2030]
colors = ['#ff9999', '#66b3ff']

plt.figure(figsize=(8, 8))
_, _, autopct_labels = plt.pie(averages_2030, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'linewidth': 3, 'edgecolor': 'white'})

# Increase font size for pie chart labels
for label in autopct_labels:
    label.set_fontsize(16)

plt.title('Average Electricity Generation in 2030', fontsize=25, fontname='Calibri')
plt.axis('equal')
plt.tight_layout()

# Create legend
legend_labels = ['Non-Renewables', 'Renewables']
legend_colors = ['#ff9999', '#66b3ff']
plt.legend(labels=legend_labels, loc='upper left', title='Electricity Source', prop={'size': 12})

plt.show()


