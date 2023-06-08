# %%
import pandas as pd
import matplotlib.pyplot as plt

energy_capacity_per_year = pd.read_csv('electricity_generation.csv')

# %%
print(energy_capacity_per_year)

# %%
# Convert 'Year End' column to datetime object and set it as the index
energy_capacity_per_year['Year End'] = pd.to_datetime(energy_capacity_per_year['Year End'], format='%Y')
energy_capacity_per_year = energy_capacity_per_year.set_index('Year End')

# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



# Plot TSO and DSO
plt.plot(energy_capacity_per_year['TSO'], label='TSO', linewidth=3)
plt.plot(energy_capacity_per_year['DSO'], label='DSO', linewidth=3)

# Add x-axis labels
plt.xlabel('Year')

# Set the x-tick labels to every 5 years
plt.xticks(energy_capacity_per_year.index[::5], energy_capacity_per_year.index.year[::5])

# Set the y-tick interval to 200
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(200))

plt.title('Renewable Energy Generation Timeseries : TSO vs DSO')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



# Plot TSO and DSO
plt.plot(energy_capacity_per_year['Total'], label='Total Generation', linewidth=3)


# Add x-axis labels
plt.xlabel('Year')

# Set the x-tick labels to every 5 years
plt.xticks(energy_capacity_per_year.index[::5], energy_capacity_per_year.index.year[::5])

# Set the y-tick interval to 200
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(200))

plt.title('Renewable Energy Generation Timeseries')

# Add a legend
plt.legend()

# Show the plot
plt.show()


