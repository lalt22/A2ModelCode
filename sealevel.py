

import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import date
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def main():
    co2 = {}
    with open("./MATH3041/data/annual-co-emissions-by-region.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[0] == "World":
                year = int(row[2])
                co2[year] = int(row[3])

    seaLevel = {}
    with open("./MATH3041/data/climate-change_2.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[0] == "World" and row[12]:
                year = int(row[1].split("-")[0])
                data = float(row[12])
                if year in co2:
                    if year in seaLevel:
                        seaLevel[year] = (seaLevel[year][0] + 1, seaLevel[year][1] + 1)
                    else:
                        seaLevel[year] = (data,1)
    xcoord = []
    ycoord = []

    for x in co2:
        if x in seaLevel:
            xcoord.append(co2[x])

    for x in seaLevel:
        if x in co2:
            ycoord.append(seaLevel[x][0] / seaLevel[x][1])

    x = np.array(xcoord)
    y = np.array(ycoord)
    mymodel = np.poly1d(np.polyfit(x, y, 3))
    print(mymodel)
    myline = np.linspace(min(xcoord), max(xcoord), len(xcoord))
    print(myline)
    plt.scatter(xcoord, ycoord)
    plt.plot(myline, mymodel(myline), color='r')

    plt.xlabel("Average Co2 Emissions")
    plt.ylabel("Church and White Sea Level Rise (mm)")
    print(r2_score(y, mymodel(x)))
    # plt.show()
if __name__ == "__main__":
    main()