# a = [0.03796, 0.04415, 0.04641, 0.04156, 0.04065, 0.04271, 0.04807, 0.05091, 0.05487]
# b = [48.82, 62.51, 69.93, 55.69, 54.52, 60.89, 68.56, 76.05, 81.92]
import csv
import numpy as np
import matplotlib.pyplot as plt
cs = csv.reader(open('try5.csv', 'r'))
print(cs)
i = 0
I = []
V = []
for Time in cs:
    print(Time)
    I = np.append(I, Time[1])
    V = np.append(V, Time[2])
    print(i)
    i += 1
y = []
for j in range (1, i):
    y = np.append(y, float(I[j])*float(V[j]))

x = np.linspace(1, i, i-1)
plt.plot(x,y)
plt.show()