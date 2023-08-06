import pandas as pd
import numpy as np
#from scipy.integrate import simpson
import matplotlib.pyplot as plt

#Extrapolate decline and reversal from 2051 to 2100
# a = 127390708144.38707
# b = -955690500980.7921       
# fn = alog(x) + b
def extrapolate_removal(data):
    historical_emissions = data[data['Year'] <= 2050]['Annual CO₂ emissions (zero filled)']
    total_emissions = historical_emissions.sum()

    prev_emission_series = data.loc[data['Year'] == 2050, 'Annual CO₂ emissions (zero filled)']
    prev_emission = prev_emission_series.iat[0]
    
    # r = -0.32738
    print(total_emissions/1000000000)
    print(prev_emission/1000000000)

    #Average yearly emissions
    avg_emissions = total_emissions/(2100-2050+1)
    print("Average Emissions = ")
    print(avg_emissions/1000000000)
    rev_data_year = []
    rev_data_entity = []
    rev_data_emission = []
    for y in range(2050, 2100+1):
        rev_data_entity.append('World')
        rev_data_year.append(y)
        rev_data_emission.append(avg_emissions*-1)
    new_df = pd.DataFrame({'Entity': rev_data_entity, 'Year': rev_data_year, 'Annual CO₂ emissions (zero filled)': rev_data_emission})
    del rev_data_emission, rev_data_entity, rev_data_year
    return new_df


#Extrapolate yearly decline from 2022-2050 s.t emissions end at 0
#Following exponential decay r = 0.442227 - multiply emissions by 1-r
def extrapolate_decline(data):
    prev_emission_series = data.loc[data['Year'] == 2021, 'Annual CO₂ emissions (zero filled)']
    prev_emission = prev_emission_series.iat[0]
    r = 0.442227
    print(prev_emission/1000000000)
    decl_data_year = []
    decl_data_entity = []
    decl_data_emission = []
    for y in range(2022, 2050 + 1):
        cur_emission = prev_emission*(1-r)
        decl_data_entity.append('World')
        decl_data_year.append(y)
        decl_data_emission.append(cur_emission)
        prev_emission = cur_emission

    new_df = pd.DataFrame({'Entity': decl_data_entity, 'Year': decl_data_year, 'Annual CO₂ emissions (zero filled)': decl_data_emission})
    del decl_data_emission, decl_data_entity, decl_data_year
    return new_df

data_url = "https://raw.githubusercontent.com/janzika/MATH3041/618718923297ec99e11cd3cd25af5ba1173d4a33/data/annual-co-emissions-by-region.csv"
climate_df = pd.read_csv('https://raw.githubusercontent.com/janzika/MATH3041/618718923297ec99e11cd3cd25af5ba1173d4a33/data/climate-change_2.csv')


data = pd.read_csv(data_url)

#Get world data
req_column = 'World'
world_df = data.query('Entity == @req_column')
climate_df = climate_df[['Entity', 'Annual average']]
climate_data = climate_df.query('Entity == @req_column')

climate_data = climate_data.dropna()
climate_data = climate_data.drop_duplicates()

world_data = world_df[['Entity', 'Year', 'Annual CO₂ emissions (zero filled)']]
sum = world_data.sum()
print("Historic sum=", sum)
#Drop values before 2000
# world_data = world_data.drop(world_data.index[:251])


#First, get declining data and add to DF
world_data_decl = extrapolate_decline(world_data)
world_data = world_data._append(world_data_decl, ignore_index = True)
sum = world_data.sum()
print("Decline sum=", sum)

x_tmp = world_data['Year']
y_tmp = world_data['Annual CO₂ emissions (zero filled)']

t_tmp = np.log(x_tmp)
a,b = np.polyfit(t_tmp, y_tmp, 1)
print("a =", a, "b= ",b)


#Next get reversal data and add to DF
world_data_rev = extrapolate_removal(world_data)
world_data = world_data._append(world_data_rev, ignore_index = True)
print(world_data.to_string())
sum = world_data.sum()
print("Removal sum=", sum)



fig1 = plt.figure(figsize=(10,6))
x = world_data['Year']
y = (world_data['Annual CO₂ emissions (zero filled)'])/1000000000

# a, b = np.polyfit(x, y, 1)
print(y)
print(y.to_string())
plt.subplot
plt.plot(x, y, color='green')
# plt.plot(x_tmp, a*x_tmp + b, color='yellow', label='Line of Best Fit')
a_2023_50 = np.trapz(x[273:300], dx = 50)
a_2050_100 = np.trapz(x[301:351], dx = 50)
print("Area=", a_2023_50, "Area2=", a_2050_100)
plt.xlabel('Year')
plt.ylabel('Annual CO₂ emissions')
plt.title("World Carbon Emissions From 1750-2100 - Net Zero Atmospheric Carbon")

# fig2 = plt.figure(figsize=(10, 6))
# x = world_data['Annual CO₂ emissions (zero filled)']/1000000000
# y = climate_data['Annual average']
# plt.scatter(x, y)
plt.show()

