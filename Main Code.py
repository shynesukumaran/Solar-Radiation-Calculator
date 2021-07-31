#Load libraries

import pandas as pd
import numpy as np
import glob
import math
import pvlib
import matplotlib.pyplot as plt
import math
import requests
import urllib3
import csv
from datetime import datetime
import datetime as dt
import html5lib
import requests
import openpyxl
import json
import time, threading


response = requests.get("*API CALL LINK*")
print(response.status_code)
data = response.json()
print(data)



test_weather = pd.DataFrame(dtype = float)
header = ["Date and Time","Wind_speed","Air_temperature","Relative_humidity","Air_pressure","Cloud_cover"]
temperature = []
pressure = []
humidity = []
wind_speed = []
cloud_cover=[]
date = []
sunset = []
sunrise = []

for i in range(len(data.get('hourly'))):
    
    temperature.append(data.get('hourly')[i].get('temp'))
    pressure.append(data.get('hourly')[i].get('pressure'))
    humidity.append(data.get('hourly')[i].get('humidity'))
    wind_speed.append(data.get('hourly')[i].get('wind_speed'))
    cloud_cover.append(data.get('hourly')[i].get('clouds'))
    timestamp=data.get('hourly')[i].get('dt')
    date_time=datetime.fromtimestamp(timestamp)
    date.append(date_time)
    
for j in range(len(data.get('daily'))): 
    sr_time=data.get('daily')[j].get('sunrise')
    date_time1=datetime.fromtimestamp(sr_time)
    sunrise.append(date_time1)
    
    ss_time=data.get('daily')[j].get('sunset')
    date_time2=datetime.fromtimestamp(ss_time)
    sunset.append(date_time2)  

    
header2 = ["SunRise_time","SunSet_time"]    
sun_data = pd.DataFrame(dtype = float) 
sun_data[header2[0]]= sunrise
sun_data[header2[1]]= sunset    
test_weather[header[0]] = date
test_weather[header[1]] = wind_speed
test_weather[header[2]] = temperature
test_weather[header[3]] = humidity
test_weather[header[4]] = pressure
test_weather[header[5]] = cloud_cover
    
test_weather['Cloud_cover']=test_weather['Cloud_cover']*(0.01)
print(test_weather)
print(sun_data)



sunrise= sun_data['SunRise_time'][0]
sunrise= pd.to_datetime(sunrise,dayfirst=True)
sunrise1= sun_data['SunRise_time'][1]
sunrise1= pd.to_datetime(sunrise1,dayfirst=True)
sunrise2= sun_data['SunRise_time'][2]
sunrise2= pd.to_datetime(sunrise2,dayfirst=True)
sunset=sun_data['SunSet_time'][0]
sunset=pd.to_datetime(sunset,dayfirst=True)
sunset1=sun_data['SunSet_time'][1]
sunset1=pd.to_datetime(sunset1,dayfirst=True)
sunset2=sun_data['SunSet_time'][2]
sunset2=pd.to_datetime(sunset2,dayfirst=True)
print(sunrise,sunrise1,sunrise2)
print(sunset,sunset1,sunset2)



forecast_data = test_weather
forecast_data['Local_time']=forecast_data['Date and Time'].dt.time
forecast_data


forecast_data['Date and Time']=pd.to_datetime(forecast_data['Date and Time'],dayfirst=True)
forecast_data = forecast_data.set_index(['Date and Time'])
forecast_data


forecast_data1 = forecast_data.loc['2021-07-27 00:00':'2021-07-27 23:00']
forecast_data1 = forecast_data1.reset_index()
forecast_data2 = forecast_data.loc['2021-07-28 00:00':'2021-07-28 23:00']
forecast_data2 = forecast_data2.reset_index()
forecast_data3 = forecast_data.loc['2021-07-29 00:00':'2021-07-29 23:00']
forecast_data3 = forecast_data3.reset_index()
forecast_data  = forecast_data.reset_index()


def Solar_Irradiance(forecast,day,longi,lati,sunrt,sunst):
    
    u = 2
    r0 = 0.97
    r1 = 0.99
    rk = 1.02
    A = 72
    a0 = (r0)*(0.4237-0.00821*((6-A)*(6-A)))
    a1 = (r1)*(0.5055+0.00595*((6.5-A)*(6.5-A)))
    k = (rk)*(0.2711+0.01858*((2.5-A)*(2.5-A)))
       
    #Go = (1367)*(1 + ((0.033)*(math.degrees(math.cos((360/365)*day)))))                 
    B = (day-81)*360/364
    DA = (23.45)*math.sin(math.radians((360*(day+284)/365)))
    #DA = pvlib.solarposition.declination_spencer71(day)*(180/(math.pi))
    print(DA)
    #EOT = pvlib.solarposition.equation_of_time_spencer71(day)
    EOT = 9.87*math.sin(math.radians(2*B))-7.53*math.cos(math.radians(B))-1.5*math.sin(math.radians(B))  
    lstm = (15*u)
    change = ((-4)*(lstm-longi))+EOT-60
    delta1 = dt.timedelta(minutes = change)
   
    
    for i,(cc,lt) in enumerate(zip(forecast['Cloud_cover'],forecast['Date and Time'])):
                                                   
        forecast.loc[i,'LST']= lt + delta1
        #forecast.loc[i,'HRA']=((forecast.loc[i,'LST']).hour-12)*15
        forecast.loc[i,'HRA']=((forecast.loc[i,'LST'].hour-12)*60 + (forecast.loc[i,'LST'].minute))*15/60
        
        forecast.loc[i,'Elevation']=math.degrees(math.asin((math.sin(math.radians(DA)))*(math.sin(math.radians(lati)))+((math.cos(math.radians(DA)))*(math.cos(math.radians(lati)))*(math.cos(math.radians(forecast.loc[i,'HRA']))))))
        forecast.loc[i,'Zenith']=math.degrees(math.acos((math.sin(math.radians(DA)))*(math.sin(math.radians(lati)))+((math.cos(math.radians(DA)))*(math.cos(math.radians(lati)))*(math.cos(math.radians(forecast.loc[i,'HRA']))))))
        #forecast.loc[i,'Elevation']=math.degrees(math.asin(((math.sin(DA))*(math.sin(lati)))+(((math.cos(DA))*(math.cos(lati))*(math.cos(forecast.loc[i,'HRA']))))))
        #forecast.loc[i,'Zenith']=math.degrees(math.acos((math.cos(lati))*(math.cos(DA))*(math.cos(forecast.loc[i,'HRA']))+(math.sin(lati))*(math.sin(DA))))
       
                
        #Tb = a0 + a1*(math.exp((-k)/math.cos(math.radians(forecast.loc[i,'Zenith']))))
        #Td = 0.271 - 0.294*Tb
        
        
        
        
        if(lt < sunrt or lt > sunst):
            forecast.loc[i,'Radiation']= 0
        else:
            
            if(forecast.loc[i,'Zenith']>90):            
                forecast.loc[i,'Radiation']=(-1)*(1367)*(1+0.033*math.cos(math.radians(360*day/365)))*(math.cos(math.radians(forecast.loc[i,'Zenith'])))*(1-((0.75)*((cc)*(cc)*(cc))))
            else:
                forecast.loc[i,'Radiation']=(1367)*(1+0.033*math.cos(math.radians(360*day/365)))*(math.cos(math.radians(forecast.loc[i,'Zenith'])))*(1-((0.75)*((cc)*(cc)*(cc))))  
                
    return forecast 
  
  
 
lat = 51.66393
long = 16.139383


d = 208
final_forecast1= Solar_Irradiance(forecast_data1,d,long,lat,sunrise,sunset)
final_forecast1



d = 208
final_forecast1= Solar_Irradiance(forecast_data1,d,long,lat,sunrise,sunset)
final_forecast1
d = 209
final_forecast2= Solar_Irradiance(forecast_data2,d,long,lat,sunrise1,sunset1)
final_forecast2
d = 210
final_forecast3= Solar_Irradiance(forecast_data3,d,long,lat,sunrise2,sunset2)
final_forecast3


df_combined = pd.concat([final_forecast1, final_forecast2,final_forecast3])
df_combined = df_combined.reset_index(drop=True)
df_combined



plt.figure(figsize=(15,10))
ax = plt.gca()
df_combined.plot(kind='line',x='Date and Time',y='Radiation',color = 'green', ax=ax)
plt.title('Calculated radiation with cloud coverage',fontsize=16)
plt.ylabel('Radiation in Watts/m2', fontsize = 12)
plt.legend(loc='upper right')
plt.xlabel('Time', fontsize = 12)
plt.gcf().autofmt_xdate()
plt.show()




