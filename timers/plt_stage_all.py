import matplotlib.pyplot as plt

f1 = open("./resultData_HDRL/HRL_stage.txt", "r")
f2 = open("./resultData_DDQN_2/HRL_stage.txt", "r")
f4 = open("./resultData_DQN/HRL_stage.txt", "r")
times1 = []
time_avg1 = []
for t in f1.readlines():
    if float(t) > 0:
        times1.append(float(t))
count = 1
for t in times1:
    tavg = sum(times1[:count])/count
    time_avg1.append(tavg)
    count += 1
tavg1 = sum(times1)/count
time_avg1.append(tavg1)


times2 = []
time_avg2 = []
for t in f2.readlines():
    if float(t) > 0:
        times2.append(float(t))
count = 1
for t in times2:
    tavg = sum(times2[:count])/count
    time_avg2.append(tavg)
    count += 1
tavg2 = sum(times2)/count
time_avg2.append(tavg2)


# times3 = []
# time_avg3 = []
# for t in f3.readlines():
#     if float(t) > 0:
#         times3.append(float(t))
# count = 1
# for t in times3:
#     tavg = sum(times3[:count])/count
#     time_avg3.append(tavg)
#     count += 1
# tavg3 = sum(times3)/count
# time_avg3.append(tavg3)
# print(len(time_avg1))
# print(len(time_avg3))

times4 = []
time_avg4 = []
for t in f4.readlines():
    if float(t) > 0:
        times4.append(float(t))
count = 1
for t in times4:
    tavg = sum(times4[:count])/count
    time_avg4.append(tavg)
    count += 1
tavg4 = sum(times4)/count
time_avg4.append(tavg4)

# count = 1
# for t in times:
#     tavg = sum(times[:count])/count
#     time_avg.append(tavg)
#     count += 1


count = 1000
t = list(range(count))

# plt.plot(t[:986], time_avg4[:986], label = "DQN")
# plt.plot(t[:972], time_avg2[:972], label = "DDQN")
# plt.plot(t[:1000], time_avg1[:1000], label = "CERP")
# plt.plot(t[:924], time_avg3[:924], label = "UCB")

plt.plot(t[:924], time_avg4[:924], label = "DQN")
plt.plot(t[:924], time_avg2[:924], label = "DDQN")
plt.plot(t[:924], time_avg1[:924], label = "CERP")
# plt.plot(t[:924], time_avg3[:924], label = "UCB")

plt.legend()
plt.xlabel('Episodes', size = 16)
plt.ylabel('Probing stage', size = 16)
# plt.ylim((0.17,0.24))
plt.show()