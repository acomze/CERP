import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

LK_CI = [1.2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
LK_LPB = [0.03796, 0.04029, 0.04065, 0.04635 , 0.04271, 0.04666 ,0.04807, 0.05015, 0.05091, 0.05186, 0.05487]
LK_CR = [48.82, 51.85,54.52, 59.55, 62.51, 66.21, 68.56, 73.74, 76.05, 78.62,81.92]
LK_CF = 3.6e9

Jet_CI = [1.2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
Jet_LPB = [0.2861, 0.2910 , 0.2971, 0.3008, 0.3034, 0.3081, 0.3127, 0.3167, 0.3224, 0.3272, 0.3288]
Jet_CR = [42.85, 46.92, 50.92, 55.10, 59.15, 63.33, 67.22, 71.20, 74.87, 77.42, 78.99]
Jet_CF = 2e9

Tao_CI = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
Tao_LPB = [0.04653, 0.04832, 0.05071, 0.04930, 0.05319, 0.05231, 0.05604, 0.05698, 0.06007, 0, 0]
Tao_CR = [64.26, 65.87, 70.12, 72.92, 75.15, 78.18, 79.79, 80.76, 84.06, 45, 50]
Tao_CF = 2.4e9

En_CI = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
En_LPB = [0.03150, 0.03266, 0.03273, 0.03311, 0.03254, 0.03257, 0.03204, 0.03200, 0.03250, 0.03277, 0.03297]
En_CR = [41.62, 43.92, 48.77, 51.15, 55.24, 57.61, 61.70, 65.83, 69.67, 73.94, 78.34]
En_CF = 3.4e9

Xgw_CI = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
Xgw_LPB = [0.04412 , 0.04642, 0.04657, 0.04513, 0.04641, 0.04544, 0.04449, 0.04483, 0.04615, 0.04754, 0.04934]
Xgw_CR = [48.28 , 48.31, 51.24, 54.33, 56.64, 60.02, 64.69, 68.43, 71.45, 74.72, 77.23]
Xgw_CF = 3.4e9

Server_CI = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
Server_LPB = [0.05044, 0.05106, 0.05050, 0.04847, 0.04532, 0.04157, 0.04341, 0.04318, 0.04358, 0.04540, 0.05453, 0.4600, 0.05232]
Server_CR = [15.12, 19.61, 24.59, 29.01, 34.51, 39.84, 44.11, 50.05, 54.86, 60.62, 63.46, 60.71, 63.46]
Server_CF = 2.5e9


x1 = np.zeros(len(Tao_CI))
re1 = np.zeros(len(Tao_CI))
x2 = np.zeros(len(Tao_CI))
re2 = np.zeros(len(Tao_CI))
x3 = np.zeros(len(Tao_CI))
re3 = np.zeros(len(Tao_CI))
x4 = np.zeros(len(Tao_CI))
re4 = np.zeros(len(Tao_CI))
x5 = np.zeros(len(Tao_CI))
re5 = np.zeros(len(Tao_CI))
x6 = np.zeros(len(Tao_CI))
re6 = np.zeros(len(Tao_CI))

for i in range(len(Tao_CI)):
    re1[i] = LK_CR[i] - LK_CI[i]
    x1[i] = LK_CI[i]

    re2[i] = Jet_CR[i] - Jet_CI[i]
    x2[i] = Jet_CI[i]

    re3[i] = Tao_CR[i] - Tao_CI[i]
    x3[i] = Tao_CI[i]

    re4[i] = En_CR[i] - En_CI[i]
    x4[i] = En_CI[i]

    re5[i] = Xgw_CR[i] - Xgw_CI[i]
    x5[i] = Xgw_CI[i]

    re6[i] = Server_CR[i] - Server_CI[i]
    x6[i] = Server_CI[i]

x = x6.reshape(-1,1)
re = re6.reshape(-1,1)
model = linear_model.LinearRegression()
model.fit(x, re)
y = model.predict(x)
t1 = x[1]
t2 = x[-1]
y1 = y[1]
y2 = y[-1]
# y2 = model.predict(t2)
k, = (y2-y1)/(t2-t1)
b, = y1-k*t1
print("Server: allocated = {} * device_state[0] + {}".format(k,b))
plt.plot(x6, model.predict(x), color='lightpink', linewidth=1.5)
plt.scatter(x6, re6,color='lightpink', label = 'Server')

# print(x2)
# print(re2)
x = x2.reshape(-1,1)
re = re2.reshape(-1,1)
# print(x)
# print(re)
model = linear_model.LinearRegression()
model.fit(x, re)
y = model.predict(x)
t1 = x[1]
t2 = x[-1]
y1 = y[1]
y2 = y[-1]
# y2 = model.predict(t2)
k, = (y2-y1)/(t2-t1)
b, = y1-k*t1
print("Jet: allocated = {} * device_state[0] + {}".format(k,b))
plt.plot(x2, model.predict(x), color='green', linewidth=1.5)
plt.scatter(x2, re2, color='green',label = 'Jetson')

x = x1.reshape(-1,1)
re = re1.reshape(-1,1)
model = linear_model.LinearRegression()
model.fit(x, re)
y = model.predict(x)
t1 = x[1]
t2 = x[-1]
y1 = y[1]
y2 = y[-1]
# y2 = model.predict(t2)
k, = (y2-y1)/(t2-t1)
b, = y1-k*t1
print("LK: allocated = {} * device_state[0] + {}".format(k,b))
plt.plot(x1, model.predict(x), color='orangered', linewidth=1.5)
plt.scatter(x1, re1, color='orangered',label = 'PC1')

x = x3.reshape(-1,1)
re = re3.reshape(-1,1)
x = x[:-2]
re = re[:-2]
model = linear_model.LinearRegression()
model.fit(x, re)
y = model.predict(x)
t1 = x[1]
t2 = x[-1]
y1 = y[1]
y2 = y[-1]
# y2 = model.predict(t2)
k, = (y2-y1)/(t2-t1)
b, = y1-k*t1
print("Tao: allocated = {} * device_state[0] + {}".format(k,b))
plt.plot(x3, model.predict(x3.reshape(-1,1)), color='gold', linewidth=1.5)
plt.scatter(x3[:-2], re3[:-2], color='gold',label = 'PC2')

x = x4.reshape(-1,1)
re = re4.reshape(-1,1)
model = linear_model.LinearRegression()
model.fit(x, re)
y = model.predict(x)
t1 = x[1]
t2 = x[-1]
y1 = y[1]
y2 = y[-1]
# y2 = model.predict(t2)
k, = (y2-y1)/(t2-t1)
b, = y1-k*t1
print("En: allocated = {} * device_state[0] + {}".format(k,b))
plt.plot(x4, model.predict(x), color='deepskyblue', linewidth=1.5)
plt.scatter(x4, re4, color='deepskyblue',label = 'PC3')

x = x5.reshape(-1,1)
re = re5.reshape(-1,1)
model = linear_model.LinearRegression()
model.fit(x, re)
y = model.predict(x)
t1 = x[1]
t2 = x[-1]
y1 = y[1]
y2 = y[-1]
# y2 = model.predict(t2)
k, = (y2-y1)/(t2-t1)
b, = y1-k*t1
print("Xgw: allocated = {} * device_state[0] + {}".format(k,b))
plt.plot(x5, model.predict(x), color='orange', linewidth=1.5)
plt.scatter(x5, re5, color='orange',label = 'PC4')



# x = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# plt.scatter(x, re2, label = 'Jetson')
# plt.scatter(x, re1, label = 'PC1')
# plt.scatter(x, re3, label = 'PC2')
# plt.scatter(x, re4, label = 'PC3')
# plt.scatter(x, re5, label = 'PC4')
# plt.scatter(x, re6, label = 'Server')



plt.legend()
plt.xlabel('CPU workload before task offloading (%)', size = 16)
plt.ylabel('Allocated CPU resource (%)', size = 16)
plt.show()

