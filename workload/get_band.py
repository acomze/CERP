fr = open("report_tram_all.log", "r")
fw = open("tram_bandwidth_origin.txt", "w")
fn = open("tram_bandwidth_3.txt", "w")
up = 2000
down = 500
band_list = []
for line in fr.readlines():
    line = line.split(" ")
    # print(line[-2],line[-1])
    band = int(line[-2])/int(line[-1])
    if band != 0 :
        band_list.append(band)
        # fw.write(str(band)+"\n")
    # fw.write("\n")
min_band = min(band_list)
max_band = max(band_list)
for band in band_list:
    new_band = (band-min_band)/(max_band-min_band)*(up-down) + down
    fn.write(str(new_band)+"\n")
fr.close()
fw.close()
fn.close()