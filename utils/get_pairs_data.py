import pandas as pd
import random
import os
import sys
import numpy as np


# export MAD='/Users/foufamastafa/Documents/master_thesis_KTH/MAD_anomaly_detection'
assert os.environ.get('MAD'), 'Please set the environment variable MAD'
MAD = os.environ['MAD']
DATA_PATH = MAD + "/data/"
sys.path.append(MAD + '/data/')
sys.path.append(MAD + '/utils/')

from data import AnomalyDataset

import argparse

parser = argparse.ArgumentParser(description='Evaluating Siamese BERT on extremely-skewed dataset. ')

parser.add_argument('--nb_reference', type=int, default=1,
    help='Strategy used to compare test set with N reference normal observations. We strategy'
         'in {1,3} ')
parser.add_argument('--nb_pairs_normal', type=int, default=1e3,
    help='Number of pairs of Normal Class used for comparison in the training set ')

parser.add_argument('--percentage_anomaly', type=float, default=100,
    help='Percentage of anomalies kept in the training set. This ratio is in percentage')

parser.add_argument('--data_filename', type=str,
    help='Relative path of data ')

parser.add_argument('--train_size', type=float, default=0.5,
                    help = "Training size ratio chose for the split of data. Then test_size = 1 - train_size")


args = parser.parse_args()
NB_REFERENCE_NORMAL = args.nb_reference
N_pairs_normal = args.nb_pairs_normal
PERCENTAGE_ANOMALY = args.percentage_anomaly
SPLIT_ARR = [args.train_size, 1-args.train_size]
DATA_NAME = args.data_filename

random.seed(1995)

# df = pd.read_csv(DATA_PATH + "/df_anomaly_XLM_en_2048_embed.csv")

def get_pairs(df_normal, df_anomaly, N_pairs_normal):
    # df_merge_2: normal, normal (fraction)
    # Strategy 1
    # Take one portion of the data
    # Dublicate it
    # As for df_merge_1, build all possible pairs from it
    # Let k be the fraction of points we take from df_normal
    # If df_normal of shape n1
    # Then we expect to have n1**2/k**2 pairs
    # We can force the number of pairs we desire
    # And from it, we deduce k
    # Indeed we have n1**2/k**2 = N_pairs_normal
    # So k = np.sqrt(n1**2/N)
    n1 = df_normal.shape[0]
    k = np.sqrt(n1 ** 2 / N_pairs_normal)
    df_fraction_normal = df_normal.sample(frac=1 / k, random_state=1995)  # we have n1/k elements from this
    df_fraction_normal['key'] = 1
    df_merge_2 = pd.merge(df_fraction_normal, df_fraction_normal, on='key').drop('key', axis=1)
    df_merge_2.drop('label_normal_y', axis = 1, inplace = True)
    df_merge_2.rename(columns = {'label_normal_x':'label'}, inplace = True)

    print(df_merge_2.columns)
    # df_merge_1 : normal, anomaly
    df_anomaly['key'] = 1
    df_normal['key'] = 1
    # Instead of merging df_normal with df_anomaly which results in many pairs
    # We can merge df_fraction_normal
    df_merge_1 = pd.merge(df_fraction_normal, df_anomaly, on='key').drop('key', axis=1)
    # For a pair (normal, anomaly) the label that we give is the one from anomaly ie 1
    df_merge_1.drop('label_normal', axis=1, inplace = True)
    df_merge_1.rename(columns = {'label_anomaly': 'label'}, inplace=True)

    # Concatenate both dataframe
    df_concat = pd.concat([df_merge_1, df_merge_2], axis=0)
    df_concat = df_concat.sample(frac = 1)
    print("Fraction of normal data {}%".format((1 / k) * 100))
    # print(df_concat.sample(2))
    return df_concat

if __name__ == "__main__":
    
    # 0/ Define train and test indices and store them in a dictionary that we can use in the future
    #  Good approach for reproducibility 
    dataset = AnomalyDataset(DATA_NAME, train_size = SPLIT_ARR[0], test_size = SPLIT_ARR[1])

    df = dataset.df
    # 1/ Get training data and test data
    # @TODO: check how df_train is
    df_train, df_test, storage_indices = dataset.get_train_test(SPLIT_ARR)

    # # For good comparison, we use the indices that we stored previously in a dictionary and used for
    # # experiments in our paper. (from sentence_ROBERTA experiments)
    # with open(DATA_PATH + "/storage_indices_train_test.dic", "rb") as f:
    #     storage_indices = pickle.load(f)
    #
    # df_train, df_test = df[df.index.isin(storage_indices['train'])], df[df.index.isin(storage_indices['test'])]
    # print(storage_indices.keys())
    # print(df_train.shape, df_test.shape)


    def prepare_for_fairs(df):
        df_anomaly = df.groupby('label').get_group(dataset.int2label['anomaly']).loc[:, 'message_cleaned']
        df_normal = df.groupby('label').get_group(dataset.int2label['normal']).loc[:, 'message_cleaned']

        df_anomaly = pd.DataFrame(df_anomaly)
        df_normal = pd.DataFrame(df_normal)

        df_anomaly['label_anomaly'] = dataset.int2label['anomaly']
        df_normal['label_normal'] = dataset.int2label['normal']

        if PERCENTAGE_ANOMALY != 100: # If the percentage of desired anomalies is != 100%
            print("We take {}% of anomalies".format(PERCENTAGE_ANOMALY))
            print("Original number of anomalies {}: ".format(df_anomaly.shape[0]))
            # Then take only a fraction of anomalies = anomaly here
            ratio_anomaly = PERCENTAGE_ANOMALY/100
            nb_normal = df_normal.shape[0]
            # We know ratio_anomaly = nb_normal/(nb_normal + nb_expected_anomaly)
            # We deduce from that:
            nb_expected_anomaly = ratio_anomaly*nb_normal/(1-ratio_anomaly)
            df_anomaly = df_anomaly.sample(n = int(nb_expected_anomaly) , random_state = 1995)
            print("Sampled number of anomalies {}: ".format(df_anomaly.shape[0]))

        return df_normal, df_anomaly

    # For training, it works well but we made mistake in df_test
    # We need ot keep the indices intact when focusing on df_test
    df_normal_train, df_anomaly_train = prepare_for_fairs(df_train)

    # 2/ Get pairs_train , pairs_test
    df_concat_train = get_pairs(df_normal_train, df_anomaly_train, N_pairs_normal=N_pairs_normal)


    # 3/ Save them in data directory
    df_concat_train.to_csv(MAD + "/data/train/pairs_train.tsv", sep='\t', index=False)

    # For test set
    # Strategy 1 : only one reference observation for each x_test_obs
    if NB_REFERENCE_NORMAL == 1:
        """
        @TODO: 
    
        1/ Take reference normal from TRAINING !!!
        2/ For each observation, compare with each of the 3 reference observations 
    
        With this implementation, what we do is that we chose 3 random reference normal observations from training set
        We assign them to every test observation only once! So for every observation we have compare(x_obs, random(reference_normal))
    
        In the future, we want: most_common[ (x_obs, reference_normal(1)), (x_obs, reference_normal(2)), (x_obs, reference_normal(3)) ] 
        """
        # Prepare test data

        # file_indices_train_test = DATA_PATH + "/storage_indices_train_test.dic"
        # # Get dictionary with indices from train/test set
        # with open(file_indices_train_test, "rb") as f:
        #     storage_indices = pickle.load(f)

        # df = pd.read_csv("./data/df_anomaly_XLM_en_2048_embed.csv")
        # df_test = df[df.index.isin(storage_indices['test'])]
        # print(df_test.columns)

        # Step 2: We chose to take 3 normal representant as comparison for now
        # For each x_new , we do compare(x_new, x_normal(1)), compare(x_new, x_normal(2)) compare(x_new, x_normal(3))
        # Then we have label_1, label_2, label_3
        # label(x_new) = most_common_label(label_1, label_2, label_3)
        # Get 3 random normal representant
        df_test_sample_normal = df_test.groupby('is_anomaly').get_group(0).loc[:4, 'message_cleaned']  # 3 representant
        arr_normal_repr = np.array(df_test_sample_normal.values)
        # print (arr_normal_repr.shape[0])
        N_rep = df_test.shape[0] // arr_normal_repr.shape[0]
        # expand array of reference representants of normal
        arr_normal_repr_expand = np.tile(arr_normal_repr, N_rep)
        while arr_normal_repr_expand.shape[0] != df_test.shape[0]:
            arr_normal_repr_expand = np.append(arr_normal_repr_expand, [arr_normal_repr_expand[0]], axis=0)
        assert arr_normal_repr_expand.shape[0] == df_test.shape[
            0], "The reference normal texts does not match the test dataframe"

        # Concatenate df_test with arr_normal_repr_expand
        """
        @TODO: normally we should map 0 --> 1 and 1 --> 0 in this labelling because the model predicts the exact opposite from Pearson Correlation = -1 
    
        In the future I will do in the train set : label_true(normal, normal) = 1 and label_true(normal, anomaly) = 0
        But now, label_true(normal, normal) = 1 and label_true(normal, anomaly) = 0
    
        ------------------------------------------
        Let's test now with: 
        The current labelling where (anomaly, normal_reference ) will be label(anomaly) = 1 
        And (normal, anomaly) = 0 
        """
        df_test = df_test.loc[:, ['message_cleaned', "is_anomaly"]]
        df_test = pd.DataFrame(df_test)
        df_test['reference_normal'] = arr_normal_repr_expand
        # print(df_test.sample(2))

        df_test = df_test.reset_index(drop=True)
        df_test.to_csv(DATA_PATH + "/test/pairs_test.tsv", sep="\t")

    if NB_REFERENCE_NORMAL == 3:
        """
        In the following,

        We do exactly the same as before except that now we consider 3 reference comparisons instead of 1 
        and we get the most common label as our predicted_label
        """

        # Step 2: We chose to take 3 normal representant as comparison for now
        # For each x_new , we do compare(x_new, x_normal(1)), compare(x_new, x_normal(2)) compare(x_new, x_normal(3))
        # Then we have label_1, label_2, label_3
        # label(x_new) = most_common_label(label_1, label_2, label_3)
        # Get 3 random normal representant

        # @TODO: IN the future, you can get 3 representant from the TRAINING DATA not the TEST DATA as we do now.
        df_test_sample_normal = df_test.groupby('is_anomaly').get_group(0).loc[:4,
                                 'message_cleaned']  # get 3 representant
        arr_normal_repr = np.array(df_test_sample_normal.values)


        # We want [x_reference_normal_1 for _ in range(N_test_obs)] , [x_reference_normal_2 for _ in range(N_test_obs)], [x_reference_normal_3 for _ in range(N_test_obs)]
        N_test_obs = df_test.shape[0]
        ref_1_arr, ref_2_arr, ref_3_arr = [arr_normal_repr[0] for _ in range(N_test_obs)], [arr_normal_repr[1] for _
                                                                                             in range(N_test_obs)], [
                                              arr_normal_repr[2] for _ in range(N_test_obs)]
        ref_arr_tot = ref_1_arr + ref_2_arr + ref_3_arr  # concatenate above arrays

        # We extend df_test 3 times : [df_test, df_test, df_test]
        df_test_expand = pd.concat([df_test] * 3)  # Keep the index intact
        # Add a new columb called 'reference_obs_normal' with reference observations (from normal in this case)
        df_test_expand['reference_normal'] = ref_arr_tot

        df_test_expand.to_csv(DATA_PATH + "/test/pairs_test.tsv", sep="\t")

