'''
Use a series of significance test to compare the differences between features.
Also, Between participants.
Then, we can see if there is some features of participants performed same in
biological signals no matter fear or no-fear.
'''
import numpy as np
import scipy.stats as stats
from Abnormal_analyze.Abnormal_extraction import data_prepare


def get_features_differences(features_array, features_name):
    # Compare the differences between 0 and 1 in different features
    features_array = features_array.transpose()
    fear_array = np.zeros(features_array.shape[1])
    non_fear_array = np.zeros(features_array.shape[1])
    for index_f, frame in enumerate(features_array):
        # Split feature into two groups: 0 and 1
        if int(frame[123]) == 0:
            non_fear_array = np.vstack((non_fear_array, frame))
        elif int(frame[123]) == 1:
            fear_array = np.vstack((fear_array, frame))
    non_fear_array = np.delete(non_fear_array, 0, axis=0)
    fear_array = np.delete(fear_array, 0, axis=0)
    non_fear_array = non_fear_array.transpose()
    fear_array = fear_array.transpose()
    # Generating all the result of significance test in different functions
    significance_test_dic = {}
    for index in range(len(fear_array) - 3):
        feature_name = features_name[index]
        fear_feature = fear_array[index]
        non_fear_feature = non_fear_array[index]
        Kolmogorov_Smirnov_result_t, Kolmogorov_Smirnov_result_p = stats.ks_2samp(fear_feature, non_fear_feature)
        Kruskal_Wallis_result_t, Kruskal_Wallis_result_p = stats.kruskal(fear_feature, non_fear_feature)
        Mann_Whitney_result_t, Mann_Whitney_result_p = stats.mannwhitneyu(fear_feature, non_fear_feature)
        significance_test_dic[feature_name] = {'Kolmogorov_Smirnov': Kolmogorov_Smirnov_result_p,
                                               'Kruskal_Wallis': Kruskal_Wallis_result_p,
                                               'Mann_Whitney': Mann_Whitney_result_p}
    return significance_test_dic


def get_features_sig(significance_dic):
    features_sig = {}
    for feature in significance_dic.items():
        if feature[1]['Kruskal_Wallis'] < 0.05 or feature[1]['Mann_Whitney'] < 0.05:
            features_sig[feature[0]] = min(feature[1]['Kruskal_Wallis'], feature[1]['Mann_Whitney'])
    return features_sig


def get_features_differences_P(features_array, features_name):
    # Compare the differences between 0 and 1 in different features of participants
    # Split data according to Participant
    features_array = features_array.transpose()
    all_P_list = []
    all_P_num = []
    now_P = 0
    for index_f, frame in enumerate(features_array):
        if int(frame[124]) != now_P and index_f == 0:
            now_P = int(frame[124])
            now_P_array = np.zeros(features_array.shape[1])
            now_P_array = np.vstack((now_P_array, frame))
        elif int(frame[124]) != now_P and index_f != 0:
            now_P_array = np.delete(now_P_array, 0, axis=0)
            all_P_list.append(now_P_array)
            all_P_num.append(now_P)
            now_P = int(frame[124])
            now_P_array = np.zeros(features_array.shape[1])
            now_P_array = np.vstack((now_P_array, frame))
        elif int(frame[124]) == now_P:
            now_P_array = np.vstack((now_P_array, frame))
    # Put the last one into the list
    all_P_num.append(now_P)
    all_P_list.append(now_P_array)
    # generate all the result of significance test
    P_significance_test_dic_all = {}
    for index, participant in enumerate(all_P_list):
        participant = participant.transpose()
        P_significance_test_dic = get_features_differences(participant, features_name)
        P_significance_test_dic_all[all_P_num[index]] = P_significance_test_dic

    return P_significance_test_dic_all


def get_features_sig_P(significance_dic_P):
    features_sig_P = {}
    for participant in significance_dic_P.items():
        features_sig = {}
        for feature in participant[1].items():
            if feature[1]['Kruskal_Wallis'] < 0.05 or feature[1]['Mann_Whitney'] < 0.05:
                features_sig[feature[0]] = min(feature[1]['Kruskal_Wallis'], feature[1]['Mann_Whitney'])
        features_sig_P[participant[0]] = features_sig
    return features_sig_P


def get_firstX_score_set(features_sig):
    # Get the former sorted features and scores in whole dataset
    sorted_features_sig = sorted(features_sig.items(), key=lambda kv: (kv[1], kv[0]))
    return sorted_features_sig


def get_firstX_score_P(features_sig_P, F_number=10):
    # Get the former X features and scores in each participant
    sorted_features_sig_P = {}
    for participant in features_sig_P.items():
        sorted_features_sig = sorted(participant[1].items(), key=lambda kv: (kv[1], kv[0]))
        F_name, F_sig = zip(*sorted_features_sig[:F_number])
        tuple_mean = ('P_sig_mean', np.mean(F_sig))
        tuple_std = ('P_sig_std', np.std(F_sig))
        tuple_median = ('P_sig_median', np.median(F_sig))
        Former_X_sig = sorted_features_sig[:F_number]
        Former_X_sig.append(tuple_mean)
        Former_X_sig.append(tuple_std)
        Former_X_sig.append(tuple_median)
        sorted_features_sig_P[participant[0]] = Former_X_sig
    return sorted_features_sig_P

def get_sorted_features_sig(json_file):
    train_array_all, test_array_all, test_feature_name_list = data_prepare(
        json_file=json_file)

    # concat train and test
    array_all = np.hstack((train_array_all, test_array_all))

    significance_test_dic = get_features_differences(array_all, test_feature_name_list)
    features_sig = get_features_sig(significance_test_dic)

    # get the first 10, 20, 30 features which have the lowest Significance score.
    # For whole part of dataset.
    sorted_features_sig = get_firstX_score_set(features_sig)
    return sorted_features_sig

def transfer_P_sigTest(significance_dic_P):
    features_sig_P_trans = {}
    for participant in significance_dic_P.items():
        features_sig = {}
        for feature in participant[1].items():
            features_sig[feature[0]] = min(feature[1]['Kruskal_Wallis'], feature[1]['Mann_Whitney'])
        features_sig_P_trans[participant[0]] = features_sig
    return features_sig_P_trans

def get_sorted_features_sig_P(json_file, F_number):
    train_array_all, test_array_all, test_feature_name_list = data_prepare(
        json_file=json_file)

    # concat train and test
    array_all = np.hstack((train_array_all, test_array_all))

    P_significance_test_dic = get_features_differences_P(array_all, test_feature_name_list)
    features_sig_P_trans = transfer_P_sigTest(P_significance_test_dic)

    # get the first 10, 20, 30 features which have the lowest Significance score for each participant.
    # For whole part of dataset.
    sorted_features_sig_P = get_firstX_score_P(features_sig_P_trans, F_number=F_number)
    return sorted_features_sig_P, test_feature_name_list

if __name__ == '__main__':
    train_array_all, test_array_all, test_feature_name_list = data_prepare(
        json_file='../Feature_wise_Normalization/Pack_all_data.json')

    # concat train and test
    array_all = np.hstack((train_array_all, test_array_all))

    significance_test_dic = get_features_differences(array_all, test_feature_name_list)
    features_sig = get_features_sig(significance_test_dic)
    # print(features_sig)
    # print(len(features_sig))

    # get the first 10, 20, 30 features which have the lowest Significance score.
    # For whole part of dataset.
    sorted_features_sig = get_firstX_score_set(features_sig)

    P_significance_test_dic = get_features_differences_P(array_all, test_feature_name_list)
    features_sig_P = get_features_sig_P(P_significance_test_dic)
    # print(features_sig_P)
    # print(len(features_sig_P))

    # get the first 10, 20, 30 features which have the lowest Significance score for each participant.
    # For whole part of dataset.
    sorted_features_sig_P = get_firstX_score_P(features_sig_P, F_number=30)

    print(1)
