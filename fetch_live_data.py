# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 23:57:21 2021

@author: sagupta2003
"""
import os
import requests
from pathlib import Path
import pandas as pd
import json
import re
import datetime


def download_all_data():
    assert "WEATHER_API_KEY" in os.environ, "missing WEATHER_API_KEY, please get one at worldweatheronline.com and store running 'export WEATHER_API_KEY=mykey'"
    api_key = os.environ["WEATHER_API_KEY"]
    URL = f"http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key={api_key}&q=Bengaluru&format=json&date=2011-01-01&enddate=2021-07-01&includelocation=yes&tp=24"
    data_dir = Path("data")
    daywise_file = data_dir / "daywisedata.csv"
    hourwise_file = data_dir / "hourwise.csv"

    whole_weather = pd.DataFrame()
    for i in range (110):
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
        whole_weather = whole_weather.append(weather, ignore_index=True)


    #Daywise data
    weather_daywise=whole_weather[["date","maxtempC","maxtempF","mintempC","mintempF","avgtempC","avgtempF"]]
    weather_daywise.to_csv(daywise_file,index=False)

    #Converting Hourly Data in normalized Form
    hourly=whole_weather['hourly']
    hourwise = pd.DataFrame(hourly.apply(lambda x:x[0]).tolist())
    cols_to_extract_value = ["weatherIconUrl", "weatherDesc"]
    for c in cols_to_extract_value:
        hourwise[c] = hourwise[c].apply(lambda x:x[0].get("value"))
    hourwise.to_csv(hourwise_file, index=False)

if __name__ == "__main__":
    download_all_data()