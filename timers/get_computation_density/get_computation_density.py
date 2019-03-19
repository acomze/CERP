import matplotlib.pyplot as plt

# CI = CPU Initial
# LPB = Latency Per image
# CR = CPU Record
# CF = CPU Frequency

image_size = 16071 * 8

LK_CI = [1.2, 5, 10, 11, 15, 20, 21, 25, 30, 31, 35, 40, 45, 50]
LK_LPB = [0.03796, 0.04029, 0.04065, 0.04156, 0.04635 , 0.04271, 0.04415, 0.04666 ,0.04807, 0.04641,  0.05015, 0.05091, 0.05186,0.05487]
LK_CR = [48.82, 51.85,54.52, 55.69, 61.55, 62.51, 60.89, 66.21, 68.56, 69.93, 73.74, 76.05, 78.62,81.92]
LK_CF = 3.6e9
# print(len(LK_CI))
# print(len(LK_LPB))
# print(len(LK_CR))

Jet_CI = [1.2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
Jet_LPB = [0.2861, 0.2910 , 0.2971, 0.3008, 0.3034, 0.3081, 0.3127, 0.3167, 0.3224, 0.3272, 0.3288]
Jet_CR = [42.85, 46.92, 50.92, 55.10, 59.15, 63.33, 67.22, 71.20, 74.87, 77.42, 78.99]
Jet_CF = 2e9

Tao_CI = [1, 5, 10, 15, 20, 25, 30, 35, 40]
Tao_LPB = [0.04653, 0.04832, 0.05071, 0.04930, 0.05319, 0.05231, 0.05604, 0.05698, 0.06007]
Tao_CR = [64.26, 65.87, 70.12, 72.92, 75.15, 78.18, 79.79, 80.76, 84.06]
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

# types = ["Server", "Desk_lk", "Desk_tao", "Jetson"]
# cpu_initial = [1.3, 1.2, 1, 1.2]
# LPB = [0.04753, 0.03797, 0.04653, 0.2861]
# CR = [29.45, 48.82, 64.26, 42.85]
# CF = [2.5e9, 3.6e9, 2.4e9, 2e9]

LK_circle_per_bit = []
for i in range(len(LK_CI)):
    cpb = LK_LPB[i]*((LK_CR[i]-LK_CI[i])/100)*LK_CF/image_size
    LK_circle_per_bit.append(cpb)

Jet_circle_per_bit = []
for i in range(len(Jet_CI)):
    cpb = Jet_LPB[i]*((Jet_CR[i]-Jet_CI[i])/100)*Jet_CF/image_size
    Jet_circle_per_bit.append(cpb)

Tao_circle_per_bit = []
for i in range(len(Tao_CI)):
    cpb = Tao_LPB[i]*((Tao_CR[i]-Tao_CI[i])/100)*Tao_CF/image_size
    Tao_circle_per_bit.append(cpb)

En_circle_per_bit = []
for i in range(len(En_CI)):
    cpb = En_LPB[i]*((En_CR[i]-En_CI[i])/100)*En_CF/image_size
    En_circle_per_bit.append(cpb)

Xgw_circle_per_bit = []
for i in range(len(Xgw_CI)):
    cpb = Xgw_LPB[i]*((Xgw_CR[i]-Xgw_CI[i])/100)*Xgw_CF/image_size
    Xgw_circle_per_bit.append(cpb)

# circle_per_bit = []
# for i in range(len(types)):
#     cpb = LPB[i]*((CR[i]-cpu_initial[i])/100)*CF[i]/image_size
#     circle_per_bit.append(cpb)

print("Desktop LK: ", LK_circle_per_bit, sum(LK_circle_per_bit)/len(LK_circle_per_bit))
print("Desktop Jetson: ", Jet_circle_per_bit, sum(Jet_circle_per_bit)/len(Jet_circle_per_bit))
print("Desktop Tao: ", Tao_circle_per_bit, sum(Tao_circle_per_bit)/len(Tao_circle_per_bit))
print("Desktop En: ", En_circle_per_bit, sum(En_circle_per_bit)/len(En_circle_per_bit))
print("Desktop Xgw: ", Xgw_circle_per_bit, sum(Xgw_circle_per_bit)/len(Xgw_circle_per_bit))
# print("Server: ", circle_per_bit[0])

# plt.plot(LK_CI, LK_circle_per_bit)
# plt.ylim(400,600)
# plt.show()
# # plt.savefig("coef_desktop.jpg")

# plt.plot(LK_CI, LK_circle_per_bit)
# plt.ylim(400,600)
# plt.show()

# plt.plot(LK_CI, LK_circle_per_bit)
# plt.ylim(400,600)
# plt.show()

# plt.plot(LK_CI, LK_circle_per_bit)
# plt.ylim(400,600)
# plt.show()

# plt.plot(LK_CI, LK_circle_per_bit)
# plt.ylim(400,600)
# plt.show()

# print("All machines: ", circle_per_bit)
# plt.plot(types, circle_per_bit)
# # plt.show()
# plt.savefig("coef_machines.jpg")