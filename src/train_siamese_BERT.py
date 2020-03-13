"""
The system trains BERT on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
NLI-like test set using N_ref comparisons between observations from the normal class.
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
import sys
import os
sys.path.append("./")
sys.path.append("../utils") # use environment variable

from EmbeddingSimilarityEvaluator import EmbeddingSimilarityEvaluatorNew
from Anomaly_Reader import AnomalyReader
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import classification_report

import argparse

parser = argparse.ArgumentParser(description='Evaluating Siamese BERT on extremely-skewed dataset. ')

parser.add_argument('--nb_reference', type=int, default=1,
    help='Strategy used to compare test set with N reference normal observations. We strategy'
        'in {1,3} ')

parser.add_argument('--epochs_train', type=int, default=1,
    help='Number of epochs to train the model ')

parser.add_argument('--is_multilingual', type=bool, default=False,
                    help = 'Set to True if you want a multilingual setting.')

args = parser.parse_args()
NB_REFERENCE_NORMAL = args.nb_reference
NB_EPOCHS = args.epochs_train
IS_MULTILINGUAL = args.is_multilingual


# export MAD='/Users/foufamastafa/Documents/master_thesis_KTH/MAD_anomaly_detection'
assert os.environ.get('MAD'), 'Please set the environment variable MAD'
MAD = os.environ['MAD']
DATA_PATH = MAD + "/data/"
OUT_PATH = MAD + "/output/"
sys.path.append(MAD + '/data/')



#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout



# Read the dataset
if IS_MULTILINGUAL:
    model_name = 'distiluse-base-multilingual-cased'
else:
    model_name = 'bert-base-uncased'
batch_size = 32
# Data in French from Flaubert github
parent_data_folder = DATA_PATH
anomaly_reader = AnomalyReader(parent_data_folder)  # after
# sts_reader = STSDataReader('../datasets/stsbenchmark')
train_num_labels = anomaly_reader.get_num_labels()
model_save_path = OUT_PATH + 'train_' + model_name + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



if not IS_MULTILINGUAL:
  # Use BERT for mapping tokens to embeddings
  word_embedding_model = models.BERT(model_name)


#####################################################################
######### Focus on transfer learning on French NLI dataset #############
#########################################################################
# Apply mean pooling to get one fixed sized sentence vector

if not IS_MULTILINGUAL: # monolingual
  pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)
  model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

else:  # multilingual
    model = SentenceTransformer(model_name)
# Convert the dataset to a DataLoader ready for training
# logging.info("Read AllNLI train dataset")
# train_data = SentencesDataset(nli_reader.get_examples('train.gz'), model=model)
# train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
# train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)

logging.info("Read NLI-like train dataset")
# get_examples just get $split as parameter
train_data = SentencesDataset(anomaly_reader.get_examples('train'), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)

print(type(train_data))



logging.info("Read ANOMALY test dataset")
# test dataset contains: 5010 rows
# valid dataset contains: 2490
# # cf $wc - l valid.x2 ==> 2490
# dev_data = SentencesDataset(examples=sts_reader.get_examples('test'), model=model)
# dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
# evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

dev_data = SentencesDataset(anomaly_reader.get_examples('test'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

# Configure the training
# @TODO: add parameters num_epochs
num_epochs = NB_EPOCHS

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))



# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=8000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )



##############################################################################
#
# Load the stored model and evaluate its performance on test data
#
##############################################################################
anomaly_reader = AnomalyReader(DATA_PATH)  # after

batch_size = 32

model = SentenceTransformer(model_save_path)
test_data = SentencesDataset(examples=anomaly_reader.get_examples("test"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluatorNew(test_dataloader)
similarity, labels = model.evaluate(evaluator)

"""
Bon normalement, 

si x > 0, alors les points sont similaires alors on a (normal, normal) alors on assigne 1 
si x < 0, alors les points sont dissimilaires alors on a (anomaly, normal) alors on assigne 0 

Mais avec notre implementation source sur SPAM, on avait dit label_true(SPAM, nonSPAM) = label_true(SPAM) = 1 (c'etait simple a coder en fait car on preserve le label du SPAM)
et label_true
"""


def threshold(x):
    if x > 0:
        # Then sentence_1 and sentence_reference_normal are SIMILAR
        return 1 # It is a normal class
    else:
        # Then sentence_1 and sentence_reference_normal are DIFFERENT
        return 0 # It is an anomaly

file_indices_train_test = DATA_PATH + "train_test_indices.dic"

labels_pred = [threshold(dot_product) for sublist in labels for dot_product in
              sublist]  # if positive value, they are similar, if negative they are dissimilar

with open(file_indices_train_test, "rb") as f:
    storage_indices = pickle.load(f)
df = pd.read_csv(DATA_PATH + "df.tsv", sep='\t')
df_test = df[df.index.isin(storage_indices['test'])]
df_test = df_test.loc[:, ["txt", "labels"]]

# @TODO: distinguish df_test which is coming from df original and the test indices
# From df_test_expand
# IMPORTANT: we need to keep the index of the observations for the group by
# So we specify index_col parameter when read_csv is called
df_test_expand = pd.read_csv(DATA_PATH + "test/pairs_test.tsv", sep = "\t", index_col = [0])
df_test_expand['labels_pred'] = labels_pred


def get_most_common_labels_to_df(df_comparisons):
    """
    Example:
    ----------------
    df_comparions:
        labels_pred
      1       1
      1       2
      2       1
      1       2

    output: pandas.DataFrame
          labels_pred
      1                 2
      2                 1

    Explanation:
    ----------------
    most_common(label_11, label_12, label_13) = most_common(1, 2, 2) = 2 for the observation with index = 1
    most_common(label_21) = most_common(1) = 1 for the observation with index = 2
    """
    print(
        "========== Estimating the labels by taking the most common labels from the comparisons to reference observations ==============")

    df_res = df_comparisons.groupby(df_comparisons.index).labels_pred.agg(pd.Series.mode)
    df_res = pd.DataFrame(df_res)
    # df_res.rename(columns = {'labels_pred': 'estimated_labels'}, inplace = True)
    return df_res


# Getting most common labels in df_res
df_res = get_most_common_labels_to_df(df_test_expand)
# classification report with sklearn comparing labels and df_test.is_spam
from sklearn.metrics import classification_report
target_names = ['anomaly', 'normal']
classification_report_df = classification_report(df_test.labels, df_res.labels_pred, target_names=target_names)
print(classification_report_df)

##############################################################################
#
# Load the stored model and evaluate its performance on Zero-Shot Data if Zero-Shot Learning
#
##############################################################################
#@TODO: add Zero-Shot parameters

ZERO_SHOT_LEARNING = True # Add it as additional parameter
DATA_PATH_MULTILINGUAL = 'multilingual/harry_potter.tok.ru'

if ZERO_SHOT_LEARNING:
    # 1/ Get data in multilingual folder
    with open(os.path.join(DATA_PATH, DATA_PATH_MULTILINGUAL),
              mode="rt", encoding="utf-8") as f:
        s_normal = [line.rstrip() for line in f if len(line.split()) > 5]  # filter out on the fly short sentences


    # @TODO: in the future, add also labels txt file in multilingual folder
    labels_arr = [1 for _ in range(len(s_normal))] # Can be changed if multilingual data is not from normal class

    # Build a test dataframe with only Russian harry potter which we expect to be predicted as Normal
    df_test_multilingual = pd.DataFrame()
    df_test_multilingual['txt'] = s_normal
    df_test_multilingual['labels'] = labels_arr

    # 2/ # Build NLI-dataset with N_ref comparisons
    # @TODO: Take your reference normal observations from Training set in the future
    # We take our representant from test data ==> Can be changed in the future
    df_test_sample_normal = df_test.groupby('labels').get_group(1).head(
            NB_REFERENCE_NORMAL).loc[:, 'txt']
    arr_normal_repr = np.array(df_test_sample_normal.values)
    # We want [x_reference_normal_1 for _ in range(N_test_obs)] , [x_reference_normal_2 for _ in range(N_test_obs)], [x_reference_normal_3 for _ in range(N_test_obs)]
    N_test_obs = df_test_multilingual.shape[0]


    def expand(txt: str, shape: int):
        """
        Example:
        --------------
        Input:
        txt = 'toto'
        shape = 3
        Output:
        ['toto', 'toto', 'toto']
        """
        arr_txt_expand = [txt for _ in range(shape)]
        return arr_txt_expand


    ref_arr_tot = []
    for txt in arr_normal_repr:
        ref_arr_tot += expand(txt, shape=N_test_obs)

    # We extend df_test 3 times : [df_test, df_test, df_test]
    df_test_expand_multilingual = pd.concat([df_test_multilingual] * NB_REFERENCE_NORMAL)  # Keep the index intact
    # Add a new columb called 'reference_obs_normal' with reference observations (from normal in this case)
    df_test_expand_multilingual['reference_normal'] = ref_arr_tot
    # df_test_expand_multilingual.drop("Unnamed: 0", axis=1, inplace=True)
    df_test_expand_multilingual.to_csv(DATA_PATH + "/multilingual/pairs_test.tsv", sep="\t")

    # Now let us test on our zero-shot data
    anomaly_reader = AnomalyReader(DATA_PATH)  # after

    batch_size = 32
    model = SentenceTransformer(model_save_path)
    test_data = SentencesDataset(examples=anomaly_reader.get_examples("zero_shot"), model=model)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    evaluator = EmbeddingSimilarityEvaluatorNew(test_dataloader)
    similarity, labels = model.evaluate(evaluator)

    labels_pred = [threshold(dot_product) for sublist in labels for dot_product in
                   sublist]

    df_test_expand_multilingual['labels_pred'] = labels_pred

    # Getting most common labels in df_res
    df_res = get_most_common_labels_to_df(df_test_expand_multilingual)
    # classification report with sklearn comparing labels and df_test.is_spam

    target_names = ['anomaly', 'normal']
    classification_report_df = classification_report([1 for _ in range(df_res.shape[0])], df_res.labels_pred,
                                                     target_names=target_names)
    print("Zero shot learning -- Classification report")
    print(classification_report_df)
