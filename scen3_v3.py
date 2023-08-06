import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_url = "https://raw.githubusercontent.com/janzika/MATH3041/618718923297ec99e11cd3cd25af5ba1173d4a33/data/annual-co-emissions-by-region.csv"
data = pd.read_csv(data_url)

#Get world data
req_column = 'World'
world_df = data.query('Entity == @req_column')
world_data = world_df[['Entity', 'Year', 'Annual CO₂ emissions (zero filled)']]

# creating projection using arctan curve
def curve(x: list):
    yvals = [0]*len(x)
    index = 0
    for val in x:
        yvals[index] = -33*np.arctan(0.3*(val-2051.3))-11
        index += 1
    return yvals

xvals = np.linspace(2021, 2100, 300)
yvals = curve(xvals)

fig1 = plt.figure(figsize=(8.6,5))
x = world_data['Year']
y = (world_data['Annual CO₂ emissions (zero filled)'])/1000000000

plt.plot(x, y, color='green')
plt.axhline(y=0, color='gray', linewidth=0.3)
plt.vlines((2050),(-65) ,(0), colors = ('gray'), linestyle=('dashed'), linewidth=0.9)
plt.vlines(2100, -60.5, 0, color='dimgrey', linestyle='solid', linewidth=1.2)
plt.plot(xvals,yvals,color='red')
plt.xlabel('Year')
plt.ylabel('Annual CO₂ emissions (Gigatonnes)')
plt.title("Integrated Net Zero Emissions Projection")

#plt.show()


# interpolation, integrals and rates of change
def f(x): return -33*np.arctan(0.3*(x-2051.3))-11

pxvals = np.linspace(2023, 2100, 500)
pyvals = curve(pxvals)
h = pxvals[2]-pxvals[1]
area = h*(pyvals[0]/2+sum(pyvals[1:len(pxvals)-1])+pyvals[-1]/2)
print(area)
print(f(2080))
print(f(2050)-f(2049))
