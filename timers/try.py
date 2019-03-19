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

#
# print("1",model.intercept_)
# print("2",model.coef_) #线性模型的系数
#
#
# # a = model.predict([[12]])
# # print(a)
# # print(-0.23811474*12 + 43.41248101)
# import matplotlib.pyplot as plt
#
# # CI = CPU Initial
# # LPB = Latency Per image
# # CR = CPU Record
# # CF = CPU Frequency
#
# image_size = 16071 * 8
#
# LK_CI = [1.2, 5, 10, 11, 15, 20, 21, 25, 30, 31, 35, 40, 50]
# LK_LPB = [0.03796, 0.04029, 0.04065, 0.04156, 0.04635 , 0.04271, 0.04415, 0.04666 ,0.04807, 0.04641,  0.05015, 0.05091, 0.05186,0.05487]
# LK_CR = [48.82, 51.85,54.52, 55.69, 61.55, 62.51, 60.89, 66.21, 68.56, 69.93, 73.74, 76.05, 78.62,81.92]
# LK_CF = 3.6e9
#
# Jet_CI = [1.2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# Jet_LPB = [0.2861, 0.2910 , 0.2971, 0.3008, 0.3034, 0.3081, 0.3127, 0.3167, 0.3224, 0.3272, 0.3288]
# Jet_CR = [42.85, 46.92, 50.92, 55.10, 59.15, 63.33, 67.22, 71.20, 74.87, 77.42, 78.99]
# Jet_CF = 2e9
#
# Tao_CI = [1, 5, 10, 15, 20, 25, 30, 35, 40]
# Tao_LPB = [0.04653, 0.04832, 0.05071, 0.04930, 0.05319, 0.05231, 0.05604, 0.05698, 0.06007]
# Tao_CR = [64.26, 65.87, 70.12, 72.92, 75.15, 78.18, 79.79, 80.76, 84.06]
# Tao_CF = 2.4e9
#
# En_CI = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# En_LPB = [0.03150, 0.03266, 0.03273, 0.03311, 0.03254, 0.03257, 0.03204, 0.03200, 0.03250, 0.03277, 0.03297]
# En_CR = [41.62, 43.92, 48.77, 51.15, 55.24, 57.61, 61.70, 65.83, 69.67, 73.94, 78.34]
# En_CF = 3.4e9
#
# Xgw_CI = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# Xgw_LPB = [0.04412 , 0.04642, 0.04657, 0.04513, 0.04641, 0.04544, 0.04449, 0.04483, 0.04615, 0.04754, 0.04934]
# Xgw_CR = [48.28 , 48.31, 51.24, 54.33, 56.64, 60.02, 64.69, 68.43, 71.45, 74.72, 77.23]
# Xgw_CF = 3.4e9
#
# Server_CI = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# Server_LPB = [0.05044, 0.05106, 0.05050, 0.04847, 0.04532, 0.04157, 0.04341, 0.04318, 0.04358, 0.04540, 0.05453, 0.4600, 0.05232]
# Server_CR = [15.12, 19.61, 24.59, 29.01, 34.51, 39.84, 44.11, 50.05, 54.86, 60.62, 63.46, 60.71, 63.46]
# Server_CF = 2.5e9
#
# # types = ["Server", "Desk_lk", "Desk_tao", "Jetson"]
# # cpu_initial = [1.3, 1.2, 1, 1.2]
# # LPB = [0.04753, 0.03797, 0.04653, 0.2861]
# # CR = [29.45, 48.82, 64.26, 42.85]
# # CF = [2.5e9, 3.6e9, 2.4e9, 2e9]
#
# LK_circle_per_bit = []
# for i in range(len(LK_CI)):
#     cpb = LK_LPB[i]*((LK_CR[i]-LK_CI[i])/100)*LK_CF/image_size
#     LK_circle_per_bit.append(cpb)
#
# Jet_circle_per_bit = []
# for i in range(len(Jet_CI)):
#     cpb = Jet_LPB[i]*((Jet_CR[i]-Jet_CI[i])/100)*Jet_CF/image_size
#     Jet_circle_per_bit.append(cpb)
#
# Tao_circle_per_bit = []
# for i in range(len(Tao_CI)):
#     cpb = Tao_LPB[i]*((Tao_CR[i]-Tao_CI[i])/100)*Tao_CF/image_size
#     Tao_circle_per_bit.append(cpb)
#
# En_circle_per_bit = []
# for i in range(len(En_CI)):
#     cpb = En_LPB[i]*((En_CR[i]-En_CI[i])/100)*En_CF/image_size
#     En_circle_per_bit.append(cpb)
#
# Xgw_circle_per_bit = []
# for i in range(len(Xgw_CI)):
#     cpb = Xgw_LPB[i]*((Xgw_CR[i]-Xgw_CI[i])/100)*Xgw_CF/image_size
#     Xgw_circle_per_bit.append(cpb)
#
# Server_circle_per_bit = []
# for i in range(len(Server_CI)):
#     cpb = Server_LPB[i]*((Server_CR[i]-Server_CI[i])/100)*Server_CF/image_size
#     Server_circle_per_bit.append(cpb)
#
# print("Desktop LK: ", LK_circle_per_bit, sum(LK_circle_per_bit)/len(LK_circle_per_bit))
# print("Desktop Jetson: ", Jet_circle_per_bit, sum(Jet_circle_per_bit)/len(Jet_circle_per_bit))
# print("Desktop Tao: ", Tao_circle_per_bit, sum(Tao_circle_per_bit)/len(Tao_circle_per_bit))
# print("Desktop En: ", En_circle_per_bit, sum(En_circle_per_bit)/len(En_circle_per_bit))
# print("Desktop Xgw: ", Xgw_circle_per_bit, sum(Xgw_circle_per_bit)/len(Xgw_circle_per_bit))
# print("Server: ", Server_circle_per_bit, sum(Server_circle_per_bit)/len(Server_circle_per_bit))
#
# # plt.plot(LK_CI, LK_circle_per_bit)
# # plt.ylim(400,600)
# # plt.show()
# # # plt.savefig("coef_desktop.jpg")
#
# # plt.plot(LK_CI, LK_circle_per_bit)
# # plt.ylim(400,600)
# # plt.show()
#
# # plt.plot(LK_CI, LK_circle_per_bit)
# # plt.ylim(400,600)
# # plt.show()
#
# # plt.plot(LK_CI, LK_circle_per_bit)
# # plt.ylim(400,600)
# # plt.show()
#
# # plt.plot(LK_CI, LK_circle_per_bit)
# # plt.ylim(400,600)
# # plt.show()
#
# # print("All machines: ", circle_per_bit)
# # plt.plot(types, circle_per_bit)
# # # plt.show()
# # plt.savefig("coef_machines.jpg")