import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_url = "https://raw.githubusercontent.com/janzika/MATH3041/618718923297ec99e11cd3cd25af5ba1173d4a33/data/annual-co-emissions-by-region.csv"
data = pd.read_csv(data_url)
req_column = 'World'
world_df = data.query('Entity == @req_column')
world_data = world_df[['Entity', 'Year', 'Annual CO₂ emissions (zero filled)']]
#world_data = world_data.drop(world_data.index[:200])
x = world_data['Year']
y = (world_data['Annual CO₂ emissions (zero filled)'])/1000000000

def curve(x):
    if x < 2030:
        return 37.1
    elif 2030 <= x < 2080:
        return 46.5/np.pi * (np.pi/2 + np.arctan(-0.12*(x-2054))) - 4.45
    else:
        return 0.1

xvals = np.linspace(2021, 2100, 100)
y_vals = [curve(xval) for xval in xvals]

plt.figure(figsize=(8.6,5))
plt.plot(x, y, color='green')
plt.plot(xvals, y_vals, color='red')
plt.vlines((2050), 0 ,curve(2050), colors = ('gray'), linestyle=('dashed'), linewidth=0.9)
plt.hlines(curve(2050), 1990, 2050, colors = 'gray', linestyle = 'dashed', linewidth = 0.9)
plt.axhline(y=0, color='black', linewidth=0.6)
plt.title('Net Zero Annual Emissions Projection')
plt.ylabel('Global emissions CO₂ (Gigatonnes)')
plt.xlabel('Year')
plt.show()

# interpolation, integrals and rates of change
pxvals = np.linspace(2050, 2080, 500)
pyvals = [curve(pxval) for pxval in pxvals]
h = pxvals[2]-pxvals[1]
area = h*(pyvals[0]/2+sum(pyvals[1:len(pxvals)-1])+pyvals[-1]/2)
print(area)
print(curve(2050))
print(curve(2021))
print(curve(2050)-curve(2049))
print((world_data.index[271] - world_data.index[251]) /20)
