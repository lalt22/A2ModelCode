import numpy as np
import matplotlib.pyplot as plt

# GDP data
gdp_data = '''World,OWID_WRL,1820,1101.5654,
World,OWID_WRL,1850,1225.081,
World,OWID_WRL,1870,1497.9823,
World,OWID_WRL,1900,2212.0432,
World,OWID_WRL,1920,2241.1655,
World,OWID_WRL,1940,3133.1982,
World,OWID_WRL,1950,3350.5664,
World,OWID_WRL,1960,4385.786,
World,OWID_WRL,1970,5951.549,
World,OWID_WRL,1980,7232.973,
World,OWID_WRL,1990,8222.479,
World,OWID_WRL,2000,9914.567,
World,OWID_WRL,2010,13179.495,
World,OWID_WRL,2016,14700.372,
World,OWID_WRL,2017,14944.094,
World,OWID_WRL,2018,15212.415'''

# CO2 emissions data
co2_data = '''World,OWID_WRL,1820,50687776
World,OWID_WRL,1850,196896030
World,OWID_WRL,1870,53253674
World,OWID_WRL,1900,1952209500
World,OWID_WRL,1920,352106240
World,OWID_WRL,1940,4854657500
World,OWID_WRL,1950,6003272000
World,OWID_WRL,1960,9387668000
World,OWID_WRL,1970,14898716000
World,OWID_WRL,1980,19501263000
World,OWID_WRL,1990,22757480000
World,OWID_WRL,2000,25453623000
World,OWID_WRL,2010,33364347000
World,OWID_WRL,2016,35524190000
World,OWID_WRL,2017,36096737000
World,OWID_WRL,2018,36826510000'''

# Convert data to NumPy arrays
gdp_data = np.genfromtxt(gdp_data.splitlines(), delimiter=',', skip_header=1, usecols=(2, 3))
co2_data = np.genfromtxt(co2_data.splitlines(), delimiter=',', skip_header=1, usecols=(2, 3))

# Calculate the coefficients for the linear regression
coefficients = np.polyfit(co2_data[:, 1], gdp_data[:, 1], 1)
slope = coefficients[0]
intercept = coefficients[1]

# Calculate the predicted values using the linear regression model
predicted_gdp = slope * co2_data[:, 1] + intercept

# Calculate R-squared (R2)
mean_gdp = np.mean(gdp_data[:, 1])
sse = np.sum((gdp_data[:, 1] - predicted_gdp) ** 2)
sst = np.sum((gdp_data[:, 1] - mean_gdp) ** 2)
r_squared = 1 - (sse / sst)

# Create scatter plot
plt.scatter(co2_data[:, 1], gdp_data[:, 1], s=50, label='Data Points')

# Plot the line of best fit
plt.plot(co2_data[:, 1], predicted_gdp, color='orange', label=f'Line of Best Fit (R2={r_squared:.2f})')

# Set plot title and labels
plt.title('CO2 Emission vs GDP')
plt.xlabel('Annual CO2 emissions')
plt.ylabel('GDP per capita')

# Display the plot
plt.legend()
plt.show()

# Print the R-squared value
print(f'R-squared = {r_squared}')
print(f'Equation of line = {equation_of_line}')
print(f'MSE = {mse}')
