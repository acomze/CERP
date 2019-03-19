import matplotlib.pyplot as plt

fr = open("./resultData_UCB/HRL_result.txt")
times = []
time_avg = []
for t in fr.readlines():
    if float(t) < 1:
        times.append(float(t))

count = 1
for t in times:
    tavg = sum(times[:count])/count
    time_avg.append(tavg)
    count += 1

tavg = sum(times)/count
time_avg.append(tavg)

t = list(range(count))
plt.plot(t, time_avg)
plt.show()