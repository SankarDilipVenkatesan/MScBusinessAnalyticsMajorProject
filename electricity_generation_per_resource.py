# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read the data from the CSV file
df = pd.read_csv("production_monthly_data.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%Y')
df.set_index('Date', inplace=True)

# Define the resources
resources = ['Coal', 'Peat', 'Oil', 'Natural Gas', 'Combustible Renewables', 'Wastes', 'Other (CHP etc)', 'Hydro', 'Wind']

# Iterate over each resource
for resource in resources:
    plt.figure(figsize=(10, 6))
    plt.plot(df[resource])
    plt.title(f'Electricity Generation in Ireland - {resource}')
    plt.xlabel('Date')
    plt.ylabel(resource)
    plt.show()

    # Perform differencing and plot the differenced series
    diff = df[resource].diff().dropna()
    plt.figure(figsize=(10, 4))
    plt.plot(diff)
    plt.title(f'Differenced {resource} Electricity Generation')
    plt.xlabel('Date')
    plt.ylabel('Difference')
    plt.show()

    # Calculate autocorrelation and partial autocorrelation plots to determine SARIMA parameters
    plot_acf(diff)
    plt.title(f'{resource} Autocorrelation')
    plt.show()

    plot_pacf(diff)
    plt.title(f'{resource} Partial Autocorrelation')
    plt.show()

    # Define the SARIMA parameters based on the plots and your domain knowledge
    order = (2, 1, 0)            # Replace p, d, q with the appropriate values
    seasonal_order = (2, 0, 2, 12)  # Replace P, D, Q, s with the appropriate values

    # Fit the SARIMA model
    model = SARIMAX(df[resource], order=order, seasonal_order=seasonal_order)
    result = model.fit()

    # Predict future values
    future_dates = pd.date_range(start='2022-11-01', end='2030-12-01', freq='MS')  # Replace the date range as needed
    predictions = result.get_forecast(steps=len(future_dates))
    predicted_values = predictions.predicted_mean

    plt.figure(figsize=(10, 6))
    plt.plot(df[resource], label='Historical')
    plt.plot(predicted_values, label='Predicted')
    plt.title(f'{resource} Electricity Generation Prediction')
    plt.xlabel('Date')
    plt.ylabel(resource)
    plt.legend()
    plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read the data from the CSV file
df = pd.read_csv("production_monthly_data.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%Y')
df.set_index('Date', inplace=True)

# Define the resources and their respective SARIMA parameters
resources = {
    'Coal': (2, 1, 2, 1, 1, 1, 12),
    'Peat': (1, 1, 3, 7, 1, 1, 12),
    'Oil': (2, 1, 2, 2, 1, 1, 12),
    'Natural Gas': (1, 0, 3, 2, 0, 3, 12),
    'Combustible Renewables': (2, 0, 3, 2, 0, 0, 12),
    'Wastes': (2, 0, 3, 2, 1, 0, 12),
    'Other (CHP etc)': (2, 0, 4, 3, 0, 1, 12),
    'Hydro': (2, 0, 4, 1, 0, 1, 12),
    'Wind': (2, 1, 0, 2, 0, 2, 12)
}

# Iterate over each resource
average_2030_values = {}
for resource, sarima_params in resources.items():
    plt.figure(figsize=(10, 6))
    plt.plot(df[resource])
    plt.title(f'Electricity Generation in Ireland - {resource}')
    plt.xlabel('Date')
    plt.ylabel(resource)
    plt.show()

    # Perform differencing and plot the differenced series
    diff = df[resource].diff().dropna()
    plt.figure(figsize=(10, 4))
    plt.plot(diff)
    plt.title(f'Differenced {resource} Electricity Generation')
    plt.xlabel('Date')
    plt.ylabel('Difference')
    plt.show()

    # Calculate autocorrelation and partial autocorrelation plots to determine SARIMA parameters
    plot_acf(diff)
    plt.title(f'{resource} Autocorrelation')
    plt.show()

    plot_pacf(diff)
    plt.title(f'{resource} Partial Autocorrelation')
    plt.show()

    # Define the SARIMA parameters based on the resource-specific parameters
    order = sarima_params[:3]
    seasonal_order = sarima_params[3:]

    # Fit the SARIMA model
    model = SARIMAX(df[resource], order=order, seasonal_order=seasonal_order)
    result = model.fit()

    # Predict future values
    future_dates = pd.date_range(start='2022-11-01', end='2030-12-01', freq='MS')  # Replace the date range as needed
    predictions = result.get_forecast(steps=len(future_dates))
    predicted_values = predictions.predicted_mean

    plt.figure(figsize=(10, 6))
    plt.plot(df[resource], label='Historical')
    plt.plot(predicted_values, label='Predicted')
    plt.title(f'{resource} Electricity Generation Prediction')
    plt.xlabel('Date')
    plt.ylabel(resource)
    plt.legend()

    # Calculate the average for the year 2030
    average_2030 = np.mean(predicted_values['2030'])

    # Print the forecasted average for the year 2030
    print(f"Forecasted average for {resource} in 2030: {average_2030}")

    # Save the predicted average of 2030 for each resource
    average_2030_values[resource] = average_2030

# Print the predicted average of 2030 for each resource
print("\nPredicted average of 2030 for each resource:")
for resource, average_2030 in average_2030_values.items():
    print(f"{resource}: {average_2030}")

# Create a pie chart of the averages
average_2030_df = pd.DataFrame.from_dict(average_2030_values, orient='index', columns=['Average 2030'])
plt.figure(figsize=(8, 8))
plt.pie(average_2030_df['Average 2030'], labels=average_2030_df.index, autopct='%1.1f%%')
plt.title('Predicted Average Electricity Generation in 2030')
plt.axis('equal')
plt.show()

# %%
# Create a pie chart of the averages
average_2030_df = pd.DataFrame.from_dict(average_2030_values, orient='index', columns=['Average 2030'])
plt.figure(figsize=(10, 8))
colors = ['#FF9999', '#FFCC99', '#FFFF99', '#CCFF99', '#99FF99', '#99FFFF', '#99CCFF', '#9999FF', '#FF99FF']
_, _, autotexts = plt.pie(average_2030_df['Average 2030'], colors=colors, startangle=90, autopct='%1.1f%%')

# Create a legend for the resources
plt.legend(average_2030_df.index, loc='upper left')

# Adjust the text size of percentage labels
for autotext in autotexts:
    autotext.set_fontsize(12)

plt.title('Predicted Average Electricity Generation in 2030', fontsize=16)
plt.axis('equal')
plt.tight_layout()
plt.show()

# %%
# Create a pie chart of the averages
average_2030_df = pd.DataFrame.from_dict(average_2030_values, orient='index', columns=['Average 2030'])
plt.figure(figsize=(10, 8))
colors = ['#FF9999', '#FFCC99', '#FFFF99', '#CCFF99', '#99FF99', '#99FFFF', '#99CCFF', '#9999FF', '#FF99FF']

# Get the index positions of "Natural Gas" and "Wind" in the DataFrame
labels = average_2030_df.index
natural_gas_index = labels.get_loc("Natural Gas")
wind_index = labels.get_loc("Wind")

# Create a list of autopct functions with empty strings except for "Natural Gas" and "Wind"
def autopct_func(value):
    if value == average_2030_df.iloc[natural_gas_index]['Average 2030']:
        return '%1.1f%%' % value
    elif value == average_2030_df.iloc[wind_index]['Average 2030']:
        return '%1.1f%%' % value
    else:
        return ''

# Create the pie chart with custom autopct functions
_, _, autotexts = plt.pie(average_2030_df['Average 2030'], colors=colors, startangle=90, autopct=autopct_func)

# Adjust the text size of percentage labels
for autotext in autotexts:
    autotext.set_fontsize(12)

# Create a legend for the resources (excluding "Natural Gas" and "Wind")
legend_labels = [label for label in labels if label not in ["Natural Gas", "Wind"]]
plt.legend(legend_labels, loc='upper left')

plt.title('Predicted Average Electricity Generation in 2030', fontsize=16)
plt.axis('equal')
plt.tight_layout()
plt.show()

# %%
# Create a DataFrame of the averages
average_2030_df = pd.DataFrame.from_dict(average_2030_values, orient='index', columns=['Average 2030'])

# Export the DataFrame to a CSV file
average_2030_df.to_csv('averages_2030.csv')

# %%
# Create a pie chart of the averages
average_2030_df = pd.DataFrame.from_dict(average_2030_values, orient='index', columns=['Average 2030'])
plt.figure(figsize=(10, 8))
colors = ['#FF9999', '#FFCC99', '#FFFF99', '#CCFF99', '#99FF99', '#99FFFF', '#99CCFF', '#9999FF', '#FF99FF']
wedges, _, autotexts = plt.pie(average_2030_df['Average 2030'], colors=colors, startangle=90, autopct='%1.1f%%', labels=average_2030_df.index)

# Create a legend for the resources
plt.legend(average_2030_df.index, loc='upper left')

# Adjust the text size of percentage labels
for autotext in autotexts:
    autotext.set_fontsize(12)

# Create lines connecting percentage labels to slices
for wedge, autotext in zip(wedges, autotexts):
    angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
    x = np.cos(np.deg2rad(angle))
    y = np.sin(np.deg2rad(angle))
    x_text = np.sign(x) * 1.2
    y_text = y * 1.2
    plt.plot([x, x_text], [y, y_text], color='black', lw=0.5)
    plt.text(x_text, y_text, autotext.get_text(), ha='center', va='center')

plt.title('Predicted Average Electricity Generation in 2030', fontsize=16)
plt.axis('equal')
plt.tight_layout()
plt.show()


# %%
# Create a pie chart of the averages
average_2030_df = pd.DataFrame.from_dict(average_2030_values, orient='index', columns=['Average 2030'])
plt.figure(figsize=(10, 8))
colors = ['#FF9999', '#FFCC99', '#FFFF99', '#CCFF99', '#99FF99', '#99FFFF', '#99CCFF', '#9999FF', '#FF99FF']

# Get the index positions of "Natural Gas" and "Wind" in the DataFrame
labels = average_2030_df.index
natural_gas_index = labels.get_loc("Natural Gas")
wind_index = labels.get_loc("Wind")

# Create a list of labels with empty strings except for "Natural Gas" and "Wind"
empty_labels = [''] * len(labels)
empty_labels[natural_gas_index] = "Natural Gas"
empty_labels[wind_index] = "Wind"

# Plot the pie chart with empty labels
wedges, _, autotexts = plt.pie(average_2030_df['Average 2030'], colors=colors, startangle=90, autopct='%1.1f%%', labels=empty_labels, pctdistance=0.85)

# Create lines connecting percentage labels to slices
for wedge, autotext in zip(wedges, autotexts):
    angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
    x = np.cos(np.deg2rad(angle))
    y = np.sin(np.deg2rad(angle))
    x_text = np.sign(x) * 1.2
    y_text = y * 1.2
    plt.plot([x, x_text], [y, y_text], color='black', lw=0.5)
    plt.text(x_text, y_text, autotext.get_text(), ha='center', va='center')

plt.title('Predicted Average Electricity Generation in 2030', fontsize=16)
plt.axis('equal')
plt.tight_layout()
plt.show()


