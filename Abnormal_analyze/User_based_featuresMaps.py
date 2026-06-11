'''
Use the result of User-based optimization to creat features maps.
'''
from random import randint

import numpy as np

from Abnormal_analyze.significance_test import get_sorted_features_sig_P

from Feature_wise_Normalization.Create_feature_maps import get_normalization, generate_feature_maps

log_file = '../Feature_wise_Normalization/log_data_normalization_functions_KNN_AD.log'
data_file = '../Feature_wise_Normalization/Pack_all_data.json'


def User_based_optimization(Feature_order, features, featureNames, strategy, F_num):
    all_feature_map, all_label = np.zeros(1), np.zeros(1)
    if strategy == 'AllFromOne':
        all_feature_map, all_label = AllFromOne_user(Feature_order, features, featureNames, F_num)
    elif strategy == 'HalfAndHalf':
        all_feature_map, all_label = HalfAndHalf(Feature_order, features, featureNames, F_num)
    # elif strategy == 'HalfAndRandom':
    #     all_feature_map, all_label = HalfAndRandom(features)
    all_feature_map[np.isnan(all_feature_map)] = 0
    all_feature_map[np.isinf(all_feature_map)] = 0
    return all_feature_map, all_label


def AllFromOne_user(Feature_order, features, featureNames, F_num):
    # Stratgy 1: Pick former Fnum features. All feature from 1 video.
    features_number = features.shape[0] - 3
    features = features.transpose()
    all_feature_map = np.zeros((F_num, F_num))
    all_label = []
    P_set = sorted(set(features[:, features_number + 1]))
    V_set = sorted(set(features[:, features_number + 2]))
    for P_num in P_set:
        Now_P_location = np.where(features[:, features_number + 1] == P_num)
        Now_P_data = features[Now_P_location[0], :]
        for V_num in V_set:
            Now_V_location = np.where(Now_P_data[:, features_number + 2] == V_num)
            Now_V_data = features[Now_V_location[0], :]
            # Pick the features for every user
            P_featureList = Feature_order[int(P_num)]
            P_feature_index_list = []
            for P_feature in P_featureList:
                if P_feature[0] in featureNames:
                    P_feature_index = featureNames.index(P_feature[0])
                    P_feature_index_list.append(P_feature_index)
            P_feature_index_list = sorted(P_feature_index_list)
            Now_V_data_picked = np.zeros(1)
            for index, P_feature_index in enumerate(P_feature_index_list):
                feature_now = Now_V_data[:, P_feature_index]
                if index == 0:
                    Now_V_data_picked = feature_now
                else:
                    Now_V_data_picked = np.vstack((Now_V_data_picked, feature_now))
            Now_V_data_picked = Now_V_data_picked.transpose()
            feature_map_resized = np.resize(Now_V_data_picked, (F_num, F_num))
            all_feature_map = np.dstack((all_feature_map, feature_map_resized))
            Now_label = Now_V_data[0][features_number]
            all_label.append(int(Now_label))
    all_feature_map = all_feature_map[:, :, 1:]
    return all_feature_map, all_label


def HalfAndHalf(Feature_order, features, featureNames, F_num):
    ## Stratgy 2: Pick former Fnum features. Half from 1 video and half from other random pariticipant but same video
    features_number = features.shape[0] - 3
    features = features.transpose()
    feature_map = np.zeros(1)
    all_feature_map = np.zeros((F_num, F_num))
    all_label = []
    P_set = sorted(set(features[:, features_number + 1]))
    V_set = sorted(set(features[:, features_number + 2]))
    for P_num in P_set:
        Now_P_location = np.where(features[:, features_number + 1] == P_num)
        Now_P_data = features[Now_P_location[0], :]
        for V_num in V_set:
            Now_V_location = np.where(Now_P_data[:, features_number + 2] == V_num)
            Now_V_data = Now_P_data[Now_V_location[0], :]
            V_label = Now_V_data[0, features_number]
            # Pick all video data in same num and in same label
            all_V_data_location = np.where((features[:, features_number + 2] == V_num) &
                                           (features[:, features_number] == V_label))
            all_V_data = features[all_V_data_location[0], :]
            P_set_sameLabel = sorted(set(all_V_data[:, features_number + 1]))
            all_V_inter = []
            for index, num in enumerate(P_set_sameLabel):
                V_same = np.where(all_V_data[:, features_number + 1] == num)
                len_index = len(V_same[0])
                all_V_inter.append(len_index)
            # integrate two data.
            Random_inter = randint(0, len(all_V_inter) - 1)
            for index in range(len(Now_V_data)):
                if index == 0:
                    feature_map = Now_V_data[index]
                    try:
                        feature_map = np.vstack((feature_map, all_V_data[index + sum(all_V_inter[:Random_inter])]))
                    except:
                        feature_map = np.vstack((feature_map, Now_V_data[index]))
                else:
                    feature_map = np.vstack((feature_map, Now_V_data[index]))
                    try:
                        feature_map = np.vstack((feature_map, all_V_data[index + sum(all_V_inter[:Random_inter])]))
                    except:
                        feature_map = np.vstack((feature_map, Now_V_data[index]))
            # Pick the features for every user
            P_featureList = Feature_order[int(P_num)]
            P_feature_index_list = []
            for P_feature in P_featureList:
                if P_feature[0] in featureNames:
                    P_feature_index = featureNames.index(P_feature[0])
                    P_feature_index_list.append(P_feature_index)
            P_feature_index_list = sorted(P_feature_index_list)
            Now_V_data_picked = np.zeros(1)
            for index, P_feature_index in enumerate(P_feature_index_list):
                feature_now = feature_map[:, P_feature_index]
                if index == 0:
                    Now_V_data_picked = feature_now
                else:
                    Now_V_data_picked = np.vstack((Now_V_data_picked, feature_now))
            Now_V_data_picked = Now_V_data_picked.transpose()
            feature_map_resized = np.resize(Now_V_data_picked, (F_num, F_num))
            all_feature_map = np.dstack((all_feature_map, feature_map_resized))
            all_label.append(V_label)
    all_feature_map = all_feature_map[:, :, 1:]
    return all_feature_map, all_label


if __name__ == '__main__':
    sorted_features_sig_P, features_name = get_sorted_features_sig_P(json_file=data_file
                                                                     , F_number=30)
    train_array_all, test_array_all = get_normalization(data_file, log_file)

    test_feature_map, test_label = User_based_optimization(Feature_order=sorted_features_sig_P, features=test_array_all,
                                                           featureNames=features_name,
                                                           strategy='AllFromOne', F_num=30)
    test_feature_map, test_label = User_based_optimization(Feature_order=sorted_features_sig_P, features=test_array_all,
                                                           featureNames=features_name,
                                                           strategy='HalfAndHalf', F_num=30)
