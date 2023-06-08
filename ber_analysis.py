# %%
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt

data = pd.read_csv('BER Public research.csv', encoding='ISO-8859-1')

# %%
data

# %%
# Select the desired variables
sample_data = data.sample(n=1000000)[['CountyName', 'Year_of_Construction', 'DeliveredEnergyMainSpace', 'DeliveredEnergySecondarySpace','EnergyRating', 'BerRating', 'CO2Rating', 'MainSpaceHeatingFuel', 'HSMainSystemEfficiency', 'DistributionLosses']]

# %%
sample_data

# %%
# Export the sample dataset to an Excel file
sample_data.to_excel('sample_dataset_BER_rating.xlsx', index=False)

# %%
# Assuming your dataset is called 'df'
sample_data['Total_Delivered_Energy'] = sample_data['DeliveredEnergyMainSpace'] + sample_data['DeliveredEnergySecondarySpace']

# %%
# Calculate the average energy rating per county
avg_energy_rating = sample_data.groupby('CountyName')['Total_Delivered_Energy'].mean().reset_index()

# Sort the counties by their average energy rating
sorted_counties = avg_energy_rating.sort_values('Total_Delivered_Energy', ascending=False)

# Print the sorted list of counties and their average energy rating
print(sorted_counties)

# %%
dublin_counties = sample_data[sample_data['CountyName'].str.contains('Dublin')]
dublin_counties_avg = dublin_counties.groupby('CountyName')['Total_Delivered_Energy'].mean().reset_index()
print(dublin_counties_avg)

# %%
# Set colour scale
colours = ['#00FF00', '#22FF00', '#44FF00', '#66FF00', '#88FF00', '#AAFF00', '#CCFF00', '#EEFF00', '#FFFF00', '#FFDD00', '#FFBB00', '#FF9900', '#FF7700', '#FF5500', '#FF3300']

# %%
# Create a dictionary that maps energy ratings to colors
energy_color_map = {
    "A1": colours[0],
    "A2": colours[1],
    "A3": colours[2],
    "B1": colours[3],
    "B2": colours[4],
    "B3": colours[5],
    "C1": colours[6],
    "C2": colours[7],
    "C3": colours[8],
    "D1": colours[9],
    "D2": colours[10],
    "E1": colours[11],
    "E2": colours[12],
    "F": colours[13],
    "G": colours[14]
}

# %%
import seaborn as sns

# Set the color palette
colors = ['#00FF00', '#22FF00', '#44FF00', '#66FF00', '#88FF00', '#AAFF00', '#CCFF00', '#EEFF00', '#FFFF00',
          '#FFDD00', '#FFBB00', '#FF9900', '#FF7700', '#FF5500', '#FF3300']
palette = sns.color_palette(colors)

# %%
# Create a scatter plot
sns.scatterplot(data=sample_data, x='BerRating', y='CO2Rating', hue='EnergyRating', palette=palette)
plt.title('Scatter plot of BER Rating vs. CO2 Rating')
plt.xlabel('BER Rating')
plt.ylabel('CO2 Rating')
plt.xlim(-200, 2000)
plt.ylim(0, 700)
plt.show()

# %%
# Set colour scale
colours = ["#00FF00", "#22FF00", "#44FF00", "#66FF00", "#88FF00", "#AAFF00", "#CCFF00", "#EEFF00", "#FFFF00", "#FFDD00", "#FFBB00", "#FF9900", "#FF7700", "#FF5500", "#FF3300"]

# Create a dictionary that maps energy ratings to colors
energy_color_map = {
    "A1": colours[0],
    "A2": colours[1],
    "A3": colours[2],
    "B1": colours[3],
    "B2": colours[4],
    "B3": colours[5],
    "C1": colours[6],
    "C2": colours[7],
    "C3": colours[8],
    "D1": colours[9],
    "D2": colours[10],
    "E1": colours[11],
    "E2": colours[12],
    "F": colours[13],
    "G": colours[14]
}

# assume the energy ratings are stored in a Series called 'energy_ratings'
dwellingcount = sample_data['EnergyRating'].value_counts().sort_index().reset_index()
dwellingcount.columns = ['EnergyRating', 'Count']

# Set the color for each energy rating

dwellingcount['Color'] = dwellingcount['EnergyRating'].apply(lambda x: energy_color_map[x.strip()])

# plot the bar chart with color
plt.bar(dwellingcount['EnergyRating'], dwellingcount['Count'], color=dwellingcount['Color'])
plt.xlabel('BER Rating')
plt.ylabel('Dwelling Count')
plt.title('Dwelling Count per BER Rating')
plt.show()

# %%
# Save the figure in PNG format with DPI of 300
plt.savefig('Dwelling Count BER Rating.png', dpi=300)

# %%
# Filter the data by year of construction and energy rating
filtered_data = sample_data[(sample_data['Year_of_Construction'] >= 1990) & (sample_data['Year_of_Construction'] <= 2022)][['Year_of_Construction', 'BerRating']]

# Group the data by year and calculate the average BER rating
grouped_data = filtered_data.groupby('Year_of_Construction').mean()

# Create a line plot of the average BER rating over time
plt.plot(grouped_data.index, grouped_data['BerRating'], linewidth=2.5)
plt.xlabel('Year of Construction')
plt.ylabel('Average BER Rating')
plt.title('Average BER Rating by Year of Construction')
plt.show()


