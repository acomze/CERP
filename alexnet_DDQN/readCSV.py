import csv

with open("sprintGo.csv",'r') as csvFile:
    reader = csv.DictReader(csvFile)
    with open("sprintGo.txt",'w') as txtFile:
        for row in reader:
            if(row['UL'] != "0"):
                txtFile.write(row['UL']+'\n')
        txtFile.close()
    csvFile.close()
