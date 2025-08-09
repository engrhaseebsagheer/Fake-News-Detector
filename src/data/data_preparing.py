import pandas as pd
import numpy as np

fake_filepath = "/Users/haseebsagheer/Documents/Python Learning/Fake-News-Detector/Data/Raw/Fake.csv"
true_filepath = "/Users/haseebsagheer/Documents/Python Learning/Fake-News-Detector/Data/Raw/True.csv"
fdf = pd.read_csv(fake_filepath)
tdf = pd.read_csv(true_filepath)
fdf["label"] = 0
tdf["label"] = 1

raw_data = pd.concat([fdf,tdf])
raw_data = raw_data.sample(frac=1,random_state = 42)
raw_data = raw_data.reset_index(drop=True)
print(raw_data.duplicated().sum())
raw_data = raw_data.drop_duplicates()
print(raw_data.duplicated().sum())
raw_data.to_csv("Clean_Data.csv",index=False)


