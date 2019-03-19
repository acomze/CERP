import random

fr = open("tram_bandwidth_3.txt", "r")
fw = open("tram_bandwidth_66.txt", "w")

band = []
for line in fr.readlines():
    band.append(line)

random.shuffle(band)

for b in band:
    fw.write(b)

fr.close()
fw.close()