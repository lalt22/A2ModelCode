import matplotlib.pyplot as plt
import numpy as np
import csv


def main():
    x = []
    for d in range(2000, 2100):
        x.append(d)
    fig, ax = plt.subplots()
    y = []
    y2 = []
    y3 = []
    # add historic data co-emissions-per-capita.csv  emissions_per_capita.csv
    with open("../data/emissions_per_capita.csv") as f :
        csv_writer = csv.reader(f, delimiter=',')
        next(csv_writer)
        for row in csv_writer:
            if row[0] == "World":
                year = int(row[2])
                if year > 2000:
                    y.append(float(row[3]))
                    y2.append(float(row[3]))
                    y3.append(float(row[3]))

    # add 2050 onwards
    b = y[-1]
    gradient = (y[-1] - y[0]) / 21
    print(gradient, b)
    print(y[-1])
    for l in range(2022, 2050):
        y.append(gradient * (l - 2022) + b)
        y2.append(gradient / 2 * (l - 2022) + b)
        y3.append(gradient / 4 * (l - 2022) + b)
    # flat from 2050
    val = y[-1]
    val2 = y2[-1]
    val3 = y3[-1]
    print(val)
    for l in range(0, 51):
        y.append(val)
        y2.append(val2)
        y3.append(val3)

    # print(y, len(y), len(x))
    ax.set_title('Carbon increasing')
    ax.plot(x, y, label = "same rate")
    ax.plot(x, y2, label = "half rate")
    ax.plot(x, y3, label = "quater rate")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()