# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 23:57:21 2021

@author: sagupta2003
"""

# from pywwo import *

import requests
import pandas as pd
import json


URL = 'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=6ae662550dfe401a9d7195421212706&q=Bengaluru&format=json&date=2011-01-01&enddate=2021-05-31&show_comments=no'


r = requests.get(URL) # get the data from the stream
val = r.json() # convert to JSON

weather=(val['data']['weather'])
weather=pd.DataFrame(weather)

#Daywise data
weather_daywise=weather[["date","maxtempC","maxtempF","mintempC","mintempF","avgtempC","avgtempF"]]
print(weather_daywise)
weather_daywise.to_excel("daywisedata.xlsx",index=False)

#Converting Hourly Data in normalized Form
hourly=weather['hourly']
df=pd.DataFrame(dict([(k,pd.Series(v)) for k,v in hourly.items()]))
hourwise=pd.json_normalize(json.loads(df.to_json(orient='records')))
print(hourwise)
hourwise.to_excel("hourwise.xlsx",index=False)

