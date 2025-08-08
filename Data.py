import pandas as pd
import numpy as np
import requests as req
#file_path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv'
#file_name = "Car_Data.csv"
#def download_file(url,filename):
#    response =  req.get(url)
#    if response.status_code == 200:
#        with open(filename, "wb") as f:
#            f.write(response.content)
#download_file(file_path,file_name)
#print(f"File has been downloaded {file_name}")
#df = pd.read_csv(file_name,header=None)
#headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
#         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
#         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
#         "peak-rpm","city-mpg","highway-mpg","price"]
#df.columns = headers
#df1 = df.replace('?',np.nan)
#df = df1.dropna(subset=["price"],axis=0)
#df.to_csv("Non_Null_Price_Data.csv",index=False)


#The Above code was written for downloading the file data and storing in a CSV
