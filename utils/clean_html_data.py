import pandas as pd
import random
import os
import sys
import numpy as np
import pickle
from bs4 import BeautifulSoup


# export MAD='/Users/foufamastafa/Documents/master_thesis_KTH/MAD_anomaly_detection'
assert os.environ.get('MAD'), 'Please set the environment variable MAD'
MAD = os.environ['MAD']
DATA_PATH = MAD + "/data/"
sys.path.append(MAD + '/data/')

# file from https://github.com/HarryPotterRus/HarryPotterrus.github.io/blob/master/index.html
file_name = DATA_PATH + "/multilingual/harry_potter_ru.html"
with open(file_name, 'rb') as f:
    soup = BeautifulSoup(f, 'html.parser')

print(type(soup.get_text()))

arr_txt = soup.get_text().strip().split("\n")
print(arr_txt[:20])
arr_txt = [txt for txt in arr_txt if len(txt)!=0]
print(arr_txt[:20])

file_name_write = DATA_PATH + "/multilingual/harry_potter_ru.txt"
with open(file_name_write, "w") as f:
    for txt in arr_txt:
        f.write(txt + " \n")