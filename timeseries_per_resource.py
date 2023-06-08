# %%
import pandas as pd
import matplotlib.pyplot as plt


production_monthly_data = pd.read_csv('production_monthly_data1.csv')

# %%
print(production_monthly_data)

# %%
production_monthly_data = production_monthly_data.rename(columns={'Unnamed: 0': 'Date'})

# %%
# Set the date column as the index
production_monthly_data = production_monthly_data.set_index('Date')

# Select columns for renewable energy fuels and non-renewable energy fuels
renewable_fuels = ['Hydro', 'Wind', 'Solar']
non_renewable_fuels = ['Coal', 'Peat', 'Oil', 'Natural Gas']

# Plot renewable energy fuels
production_monthly_data[renewable_fuels].plot(figsize=(10, 5))
plt.title('Renewable Energy Sources')
plt.xlabel('Date')
plt.ylabel('Electricity Generated (GWh)')
plt.show()

# Plot non-renewable energy fuels
production_monthly_data[non_renewable_fuels].plot(figsize=(10, 5))
plt.title('Non-Renewable Energy Sources')
plt.xlabel('Date')
plt.ylabel('Electricity Generated (GWh)')
plt.show()

# %%
print(production_monthly_data)

# %%
# Calculate the total energy output per year
total_per_year = production_monthly_data.sum(axis=1)

total_per_year = production_monthly_data.drop(['Exports', 'Imports', 'Pumping', 'Generated', 'Combustible Renewables', 'Wastes', 'Solar', 'Other (CHP etc)'], axis=1)

# Calculate the percentage of each resource per year
percent_per_year = production_monthly_data.divide(total_per_year, axis=0) * 100

# Print the percentage of each resource per year
print(percent_per_year)

# %%
import matplotlib.pyplot as plt

# Select data for 2010 and calculate the sum of each fuel source
df_2010 = total_per_year.loc['Jan-10':'Dec-10']
sums_2010 = df_2010.sum()

# Create a pie chart
plt.pie(sums_2010, labels=sums_2010.index, autopct='%1.1f%%')

# Set the title
plt.title('Fuel Sources in 2010')

# Show the chart
plt.show()

# %%
import matplotlib.pyplot as plt

# Select data for 2022 and calculate the sum of each fuel source
df_2022 = total_per_year.loc['Jan-22':'Oct-22']
sums_2022 = df_2022.sum()

# Create a pie chart
plt.pie(sums_2022, labels=sums_2022.index, autopct='%1.1f%%')

# Set the title
plt.title('Fuel Sources in 2022')

# Show the chart
plt.show()

# %%
# Select data for 2010 and calculate the sum of each fuel source
df_2010 = total_per_year.loc['Jan-10':'Dec-10']
sums_2010 = df_2010.sum()

# Define a visually appealing color scheme
colors = ['#a8dadc', '#457b9d', '#e63946', '#fca311', '#1d3557', '#8d99ae']

# Create a donut chart
fig, ax = plt.subplots()
ax.pie(sums_2010, colors=colors, autopct='%1.1f%%', startangle=90, counterclock=False, wedgeprops=dict(width=0.4, edgecolor='w'))

# Add a white circle in the middle to create a donut chart
circle = plt.Circle((0,0), 0.2, color='white')
ax.add_artist(circle)

# Add labels to the donut chart
ax.legend(sums_2010.index, loc='center', bbox_to_anchor=(1.2, 0.5), fontsize=12, title='Fuel Sources')

ax.set_title('Energy Sources in 2010', fontsize=18)

# Show the chart
plt.show()

# %%
# Select data for 2010 and calculate the sum of each fuel source
df_2010 = total_per_year.loc['Jan-10':'Dec-10']
sums_2010 = df_2010.sum()

# Define a visually appealing color scheme
colors = ['#a8dadc', '#457b9d', '#e63946', '#fca311', '#1d3557', '#8d99ae']

# Create a donut chart
fig, ax = plt.subplots()
ax.pie(sums_2010, colors=colors, startangle=90, counterclock=False, wedgeprops=dict(width=0.4, edgecolor='w'))

# Add a white circle in the middle to create a donut chart
circle = plt.Circle((0,0), 0.2, color='white')
ax.add_artist(circle)

# Add labels to the donut chart
ax.legend(sums_2010.index, loc='center', bbox_to_anchor=(1.2, 0.5), fontsize=12, title='Energy Sources')

ax.set_title('Energy Sources in 2010', fontsize=18)

# Show the chart
plt.show()

# %%
# Select data for 2022 and calculate the sum of each fuel source
df_2022 = total_per_year.loc['Jan-22':'Oct-22']
sums_2022 = df_2022.sum()

# Define a visually appealing color scheme
colors = ['#a8dadc', '#457b9d', '#e63946', '#fca311', '#1d3557', '#8d99ae']

# Create a donut chart
fig, ax = plt.subplots()
ax.pie(sums_2022, colors=colors, autopct='%1.1f%%', startangle=90, counterclock=False, wedgeprops=dict(width=0.4, edgecolor='w'))

# Add a white circle in the middle to create a donut chart
circle = plt.Circle((0,0), 0.2, color='white')
ax.add_artist(circle)

# Add labels to the donut chart
ax.legend(sums_2010.index, loc='center', bbox_to_anchor=(1.2, 0.5), fontsize=12, title='Fuel Sources')

ax.set_title('Energy Sources in 2022', fontsize=18)

# Show the chart
plt.show()

# %%
# Select data for 2022 and calculate the sum of each fuel source
df_2022 = total_per_year.loc['Jan-22':'Oct-22']
sums_2022 = df_2022.sum()

# Define a visually appealing color scheme
colors = ['#a8dadc', '#457b9d', '#e63946', '#fca311', '#1d3557', '#8d99ae']

# Create a donut chart
fig, ax = plt.subplots()
ax.pie(sums_2022, colors=colors, startangle=90, counterclock=False, wedgeprops=dict(width=0.4, edgecolor='w'))

# Add a white circle in the middle to create a donut chart
circle = plt.Circle((0,0), 0.2, color='white')
ax.add_artist(circle)

# Add labels to the donut chart
ax.legend(sums_2022.index, loc='center', bbox_to_anchor=(1.2, 0.5), fontsize=12, title='Fuel Sources')

ax.set_title('Energy Sources in 2022', fontsize=18)

# Show the chart
plt.show()


