import pandas as pd, numpy
import matplotlib.pyplot as plt

# Read the CSV files
emissions_df = pd.read_csv('https://raw.githubusercontent.com/janzika/MATH3041/618718923297ec99e11cd3cd25af5ba1173d4a33/data/annual-co-emissions-by-region.csv', sep=',')
climate_df = pd.read_csv('https://raw.githubusercontent.com/janzika/MATH3041/618718923297ec99e11cd3cd25af5ba1173d4a33/data/climate-change_2.csv')

#Only get world data
req_value = 'World'


# Extract relevant columns from emissions_df
print(emissions_df.columns)
emissions_df = emissions_df[['Entity', 'Year', 'Annual CO₂ emissions (zero filled)']]
emissions_data = emissions_df.query('Entity == @req_value')

# Extract relevant columns from climate_df
print(climate_df.columns)
climate_df = climate_df[['Entity', 'Annual average']]
climate_data = climate_df.query('Entity == @req_value')

# Merge the two dataframes on 'Entity' column - gives world data
merged_data = pd.merge(emissions_data, climate_data, on='Entity')

# Filter out rows with missing values and duplicates
climate_data = climate_data.dropna()
climate_data = climate_data.drop_duplicates()
merged_data = merged_data.dropna()

#Cut emissions data for first 44 years
emissions_data = emissions_data.drop(emissions_data.index[:44])


print(climate_data)
print(emissions_data)

fig = plt.figure(figsize=(10, 6))
x = emissions_data['Annual CO₂ emissions (zero filled)']
y = climate_data['Annual average']

t = numpy.log(x)
a, b = numpy.polyfit(t, y, 1)
x_fitted = numpy.linspace(numpy.min(x), numpy.max(x), 228)
y_fitted = a * numpy.log(x_fitted) + b
plt.scatter(x,  y)

plt.plot(x_fitted, y_fitted, color='yellow', label='Curve of Best Fit')
corr_matrix = numpy.corrcoef(numpy.log(x), x_fitted)
corr = corr_matrix[0,1]
r_2 = corr**2
print("R2=", r_2)

plt.xlabel('Annual CO₂ emissions (zero filled)')
plt.ylabel('Annual pH Measurement')
plt.title('Relationship between Annual CO₂ emissions (zero filled) and pH Measurement')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

