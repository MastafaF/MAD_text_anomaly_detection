"""
Given:
corpus_1.txt
corpus_2.txt
corpus_3.txt

We want:
pandas DataFrame
with:

txt  || labels

corpus_1   labels_of_corpus_1

corpus_2   labels_of_corpus_2

corpus_3    labels_of_corpus_3

-------------------------

We assume all the txt files stored in a directory specified by the user.
We expect user to tell us filename_normal filename of the normal class.
All other txt files will be considered as anomalous

1/ We read the txt file from class Normal --> label 1
    Build arr_sentences_normal, arr_labels_normal
2/ We read all other files as anomalous --> label 0
    Build arr_sentences_anomalous, arr_labels_anomalous
3/ Concatenate arrays
4/ Build dataframe
5/ Drop duplicates
6/ Check statistics of dataframe
7/ Build an unbalanced dataset

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import argparse
import sys



# export MAD='/Users/foufamastafa/Documents/master_thesis_KTH/MAD_anomaly_detection'
assert os.environ.get('MAD'), 'Please set the environment variable MAD'
MAD = os.environ['MAD']
DATA_PATH = MAD + "/data/"
OUT_PATH = MAD + "/output/"
sys.path.append(MAD + '/data/')


filename_normal = 'normal'
filename_anomalous = 'anomalous'



with open(os.path.join(DATA_PATH, 'processed/harry_potter.tok.en'),
          mode="rt", encoding="utf-8") as f:
    s_normal = [line.rstrip() for line in f ]


labels_normal = [1 for _ in range(len(s_normal))]


with open(os.path.join(DATA_PATH, 'processed/the_republic.tok.en'),
                       mode="rt", encoding="utf-8") as f:
    s_anomalous = [line.rstrip() for line in f ]

labels_an = [0 for _ in range(len(s_anomalous))]

s_total = s_normal + s_anomalous
labels_total = labels_normal + labels_an

df = pd.DataFrame()
df['txt'] = s_total
df['labels'] = labels_total

df.drop_duplicates(subset='txt', inplace = True)

# Some statitics on the data

def apply_len(row):
    return len(row.split())

print(apply_len("Bonjour, je m'appelle Mastafa"))

df['len'] = df['txt'].apply(apply_len)



# dataframe with short sentences:

# Filter out all short sentences:
df = df[df['len'] > 5]

# Plots of distributions of lengths for normal and NON-NORMAL data
# f, ax = plt.subplots(1, 2, figsize = (20, 6))
#
# sns.distplot(df[df["labels"] == 0]["len"], bins = 20, ax = ax[0])
# ax[0].set_xlabel("Length of normal class - Harry Potter English")
#
# sns.distplot(df[df["labels"] == 1]["len"], bins = 20, ax = ax[1])
# ax[1].set_xlabel("Length of anomalous class - The Republic English")
#
# plt.show()

# Shuffle dataframe
df = df.sample(frac = 1)

df.drop('len', axis = 1, inplace = True)

# df.to_csv(os.path.join(DATA_PATH, "processed/df.tsv"), sep = '\t')

# print(df.labels.describe())