# from pywwo import *

import requests
import json
import csv


URL = 'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=6ae662550dfe401a9d7195421212706&q=Bengaluru&format=json&date=2009-01-01&enddate=2021-05-31&show_comments=no'




r = requests.get(URL) # get the data from the stream
json_loads = r.json() # convert to JSON
file1 = open("data.txt","w") # save to file
file1.write(r.text)
# print(r.json()) 
print (type(json_loads))
print (json_loads[0])
data_file = open('data.csv', 'w', newline='')
csv_writer = csv.writer(data_file)

count = 0
for data in json_loads[0]:
    if count == 0:
        header = data.keys()
        csv_writer.writerow(header)
        count += 1
    csv_writer.writerow(data.values())
 
data_file.close()

# ini_string = json.dumps(r.text)
# print ("initial 1st dictionary", ini_string)
# print ("type of ini_object", type(ini_string))


# setKey('6ae662550dfe401a9d7195421212706', 'premium')
# w=LocalWeather('bengaluru')
# print(w.data.current_condition.temp_C)

# w=LocalWeather('sdfasdgasdga')