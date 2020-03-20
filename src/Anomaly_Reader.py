from sentence_transformers.readers import *
import csv
import gzip
import os
import pandas as pd

class AnomalyReader(object):
    """
    Reads in the SPAM Collection Dataset
    """
    def __init__(self, dataset_folder): # dataset_folder = parent folder = "whatever_is_before/data"
        self.dataset_folder = dataset_folder

    def get_examples(self, split, max_examples=0):
        """
        data_splits specified which data split to use (train, dev, test).
        """
        # s1 = open(os.path.join(self.dataset_folder, split + '.x1'),
        #                mode="rt", encoding="utf-8").readlines()
        # s2 = open(os.path.join(self.dataset_folder, split + '.x2'),
        #           mode="rt", encoding="utf-8").readlines()
        # labels = open(os.path.join(self.dataset_folder, split + '.label'),
        #                    mode="rt", encoding="utf-8").readlines()
        if split == "train":
          # Open dataframe
          df_split = pd.read_csv(os.path.join(self.dataset_folder, "train/pairs_train.tsv"), sep='\t')

          # Normal
          s1 = df_split.loc[:,'txt_x'].values
          s1 = list(s1)

          # Anomaly
          s2 = df_split.loc[:, 'txt_y'].values
          s2 = list(s2)

          # labels
          labels = df_split.loc[:, 'labels'].values
          labels = list(labels)

        # print("LABELS")
        # print(type(labels))
        # print(labels[:10])
        if split == "test":
          df_split = pd.read_csv(os.path.join(self.dataset_folder, "test/pairs_test.tsv"), sep='\t')

          # Any test observations from test set
          s1 = df_split.loc[:,'txt'].values
          s1 = list(s1)

          # reference Normal
          s2 = df_split.loc[:, 'reference_normal'].values
          s2 = list(s2)

          # labels
          labels = df_split.loc[:, 'labels'].values
          labels = list(labels)

        if split == 'zero_shot':
            df_split = pd.read_csv(os.path.join(self.dataset_folder, "zero_shot/pairs_test.tsv"), sep='\t')

            # Any test observations from test set
            s1 = df_split.loc[:, 'txt'].values
            s1 = list(s1)

            # reference Normal
            s2 = df_split.loc[:, 'reference_normal'].values
            s2 = list(s2)

            # labels
            labels = df_split.loc[:, 'labels'].values
            labels = list(labels)

        examples = []
        id = 0
        for sentence_a, sentence_b, label in zip(s1, s2, labels):
            guid = "%s-%d" % (split, id)
            id += 1
            # Each sentence is like "Lsentence"
            # So we need to take sentence[1:]
            # examples.append(InputExample(guid=guid, texts=[sentence_a[1:], sentence_b[1:]], label=self.map_label(label)))

            #When update get-data-xnli.sh for MAC, we don't have the issue with "Lsentence"
            examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b], label=label))

            if 0 < max_examples <= len(examples):
                break

        return examples

    @staticmethod
    def get_labels():
        # We did the opposite in our original implementation for SPAM Detector
        return {"same_class": 1, "opposite_class": 0}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]