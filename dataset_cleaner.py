# %%
import pandas as pd

consumption_final = pd.read_csv('consumption_industry.csv')

# %%
# transpose the rows and columns
consumption_final = consumption_final.T

# print the transposed dataset
print(consumption_final)

# %%
consumption_final = consumption_final.dropna(axis=1)

# %%
print(consumption_final)

# %%
# assuming your data is stored in a variable named 'data'
new_columns = consumption_final.iloc[0]
consumption_final = consumption_final[1:]
consumption_final = consumption_final.set_axis(new_columns, axis=1)

# %%
# Export the sample dataset to an Excel file
consumption_final.to_excel('consumption_final_prepared.xlsx', index=True)


