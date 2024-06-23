import pandas as pd
import os
import shutil

filePath = "covid-chestxray-dataset-master/metadata.csv"
imagesPath = "covid-chestxray-dataset-master/images"

df = pd.read_csv(filePath)
print(df.shape)