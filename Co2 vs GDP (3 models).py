#!/usr/bin/env python
# coding: utf-8

# In[21]:


# Linear Model

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
plt.plot(co2_data[:, 1], predicted_gdp, color='orange', label=f'Line of Best Fit')

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


# In[19]:


# Cubic Model

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

# Calculate the coefficients for the cubic regression
coefficients = np.polyfit(co2_data[:, 1], gdp_data[:, 1], 3)

# Generate a set of x-values for the cubic curve
x_values = np.linspace(min(co2_data[:, 1]), max(co2_data[:, 1]), 100)

# Calculate the predicted values using the cubic regression model
predicted_gdp = np.polyval(coefficients, x_values)

# Calculate Sum of Squared Errors (SSE)
sse = np.sum((gdp_data[:, 1] - np.polyval(coefficients, co2_data[:, 1])) ** 2)

# Calculate R-squared (R2)
mean_gdp = np.mean(gdp_data[:, 1])
sst = np.sum((gdp_data[:, 1] - mean_gdp) ** 2)
r_squared = 1 - (sse / sst)

# Create scatter plot
plt.scatter(co2_data[:, 1], gdp_data[:, 1], s=50, label='Data Points')

# Plot the cubic curve
plt.plot(x_values, predicted_gdp, color='orange', label='Cubic Curve')

# Set plot title and labels
plt.title('CO2 Emission vs GDP')
plt.xlabel('Annual CO2 emissions')
plt.ylabel('GDP per capita')

# Display the equation of the cubic curve on the plot
equation_of_curve = f'GDP per capita = {coefficients[0]} * (Annual CO2 emissions)^3 + {coefficients[1]} * (Annual CO2 emissions)^2 + {coefficients[2]} * Annual CO2 emissions + {coefficients[3]}'

# Display the plot
plt.legend()
plt.show()

print(f'R squared value = {r_squared}')
print(f'Equation of the cubic curve: {equation_of_curve}')
print(f'MSE {mse}')


# In[22]:


# Logarithmic Model

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

# Convert the CO2 emissions data to logarithmic values
log_co2_data = np.log(co2_data[:, 1])

# Define the logarithmic function
def logarithmic_function(x, a, b):
    return a * np.log(x) + b

# Fit the logarithmic function to the data
popt, _ = curve_fit(logarithmic_function, co2_data[:, 1], gdp_data[:, 1])

# Generate a set of x-values for the logarithmic curve
x_values = np.linspace(min(co2_data[:, 1]), max(co2_data[:, 1]), 100)

# Calculate the predicted values using the fitted logarithmic function
predicted_gdp = logarithmic_function(x_values, *popt)

# Create scatter plot
plt.scatter(co2_data[:, 1], gdp_data[:, 1], s=50, label='Data Points')

# Plot the logarithmic curve
plt.plot(x_values, predicted_gdp, color='orange', label='Logarithmic Curve')

# Set plot title and labels
plt.title('CO2 Emission vs GDP')
plt.xlabel('Annual CO2 emissions')
plt.ylabel('GDP per capita')

# Display the equation of the logarithmic curve on the plot
equation_of_curve = f'GDP per capita = {popt[0]} * ln(Annual CO2 emissions) + {popt[1]}'

# Display the plot
plt.legend()
plt.show()

# Calculate R-squared (R2)
residuals = gdp_data[:, 1] - logarithmic_function(co2_data[:, 1], *popt)
ssr = np.sum(residuals**2)
sst = np.sum((gdp_data[:, 1] - np.mean(gdp_data[:, 1]))**2)
r_squared = 1 - (ssr / sst)

# Calculate Mean Squared Error (MSE)
mse = np.mean(residuals**2)

print(f'R squared value = {r_squared}')
print(f'MSE = {mse}')
print(f'Equation of the logarithmic curve: {equation_of_curve}')

