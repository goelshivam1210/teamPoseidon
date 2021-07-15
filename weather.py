# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 23:57:21 2021

@author: sagupta2003
"""

# from pywwo import *

import requests
import pandas as pd
import json
import re
import datetime

URL = "http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=6ae662550dfe401a9d7195421212706&q=Bengaluru&format=json&date=2011-01-01&enddate=2021-01-01&includelocation=yes&tp=24"

whole_weather = pd.DataFrame()
for i in range (100):
    split = URL.split("&")
    subs = "date"
    res = [x for x in split if re.match(subs, x)]

    date = str((datetime.datetime(2011,1,1,0,0,0) + datetime.timedelta(days=35*i)).date())
    print ("date we are getting data from = {}".format(date))
    split[split.index(res[0])] = "date="+date
    print ("split new array = {}".format(split))
    URL = '&'.join(split)
    print ("URL = {}".format(URL))

    r = requests.get(URL) # get the data from the stream        
    val = r.json() # convert to JSON

    weather=(val['data']['weather'])
    weather=pd.DataFrame(weather)
    # print ("Weather dataframe = {}".format(weather))
    whole_weather = whole_weather.append(weather, ignore_index=True)
    # print ("whole weather = {}".format(whole_weather))

# import IPython; import sys; IPython.embed(); sys.exit(1)

#Daywise data
weather_daywise=whole_weather[["date","maxtempC","maxtempF","mintempC","mintempF","avgtempC","avgtempF"]]
print(weather_daywise)
weather_daywise.to_excel("daywisedata.xlsx",index=False)

#Converting Hourly Data in normalized Form
hourly=whole_weather['hourly']
hourwise = pd.DataFrame(hourly.apply(lambda x:x[0]).tolist())
cols_to_extract_value = ["weatherIconUrl", "weatherDesc"]
for c in cols_to_extract_value:

    hourwise[c] = hourwise[c].apply(lambda x:x[0].get("value"))
# df=pd.DataFrame(dict([(k,pd.Series(v)) for k,v in hourly.items()]))
# hourwise=pd.json_normalize(json.loads(df.to_json(orient='records')))
# print(hourwise)
hourwise.to_csv("hourwise.csv",index=False)


