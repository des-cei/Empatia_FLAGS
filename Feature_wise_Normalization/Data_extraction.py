import json
from random import shuffle

import numpy as np


def get_train_test(content):
    Participants_list = list(content.keys())
    # Randomly split participants
    n_total = len(Participants_list)
    offset = round(n_total * 0.8)
    if n_total == 0 or offset < 1:
        return [], Participants_list
    # random split the train and test
    shuffle(Participants_list)
    train_P_list = Participants_list[:offset]
    test_P_list = Participants_list[offset:]
    return train_P_list, test_P_list


def Abnormal_detection(input_array):
    # This function for filter the abnormal data
    input_array = input_array.transpose()
    output_array = np.zeros(input_array.shape[0])
    for index_f, feature in enumerate(input_array):
        Percentile = np.percentile(feature, [0, 25, 50, 75, 100])
        IQR = Percentile[3] - Percentile[1]
        UpLimit = Percentile[3] + IQR * 1.5
        DownLimit = Percentile[1] - IQR * 1.5
        # If the data more than Uplimit or less than Downlimit, see it as abnormal data
        # Use the average between former one and later one to replace it
        for index in range(len(feature) - 1):
            if (feature[index] > UpLimit) or (feature[index] < DownLimit):
                try:
                    feature[index] = ((feature[index - 1]) + (feature[index + 1])) / 2
                except:
                    continue
        if index_f == 0:
            output_array = feature
        else:
            output_array = np.vstack((output_array, feature))
    output_array = output_array.transpose()
    return output_array

def get_train_test(content, cluster_list):
    Participants_list = list(content.keys())
    Participants_list_copy = Participants_list.copy()
    # remove the users not in cluster
    for user_value in Participants_list_copy:
        int_value = int(user_value.replace('V', ''))
        if int_value not in cluster_list:
            Participants_list.remove(user_value)
    # Randomly split participants
    n_total = len(Participants_list)
    offset = round(n_total * 0.8)
    if n_total == 0 or offset < 1:
        return [], Participants_list
    # random split the train and test
    shuffle(Participants_list)
    train_P_list = Participants_list[:offset]
    test_P_list = Participants_list[offset:]
    return train_P_list, test_P_list

def get_train_test_approach1(train_cluster, test_cluster):
    train_P_list = []
    test_P_list = []
    for value in train_cluster:
        new_p = "'V" + str(value).zfill(3) + "'"
        new_p = eval(new_p)
        train_P_list.append(new_p)

    for value in test_cluster:
        new_p = "'V" + str(value).zfill(3) + "'"
        new_p = eval(new_p)
        test_P_list.append(new_p)

    # train_P_list.append(test_P_list)
    # test_P_list = sample(test_P_list, 1)
    return train_P_list, test_P_list

def data_extraction_cluster(json_file, train_cluster, test_cluster, approach_num):
    train_feature_all_dic = {}
    train_label_all_dic = {}
    test_feature_all_dic = {}
    test_label_all_dic = {}
    with open(json_file, "r", encoding="utf-8") as f:
        content = json.load(f)
    ## Split train and test based on paticipants.
    ## Warning: data from one participant should not be included both in train and test
    # train_P_list, test_P_list = get_train_test(content, cluster_list=train_cluster)
    train_cluster = train_cluster.astype(int)

    # Approach 1
    if approach_num == 'approach1':
        train_P_list, test_P_list = get_train_test_approach1(train_cluster, test_cluster)
    # Approach 2
    elif approach_num == 'approach2':
        train_P_list, test_P_list = get_train_test(content, cluster_list=(train_cluster + test_cluster))

    ## Put all the data into one ndarray
    for participant in content.items():
        for video in participant[1].items():
            for episode in video[1].values():
                if 'ReportadaBinarizado' in episode.keys():
                    for feature_name in episode.keys():
                        # Judge if the feature from the participants from train or test
                        if participant[0] in train_P_list:
                            if ('bvp' in feature_name) or ('gsr' in feature_name) or ('skt' in feature_name):
                                if feature_name not in train_feature_all_dic.keys():
                                    train_feature_all_dic[feature_name] = [episode[feature_name]]
                                else:
                                    train_feature_all_dic[feature_name].append(episode[feature_name])
                            elif ('ReportadaBinarizado' in feature_name):
                                if feature_name not in train_label_all_dic.keys():
                                    train_label_all_dic[feature_name] = [episode[feature_name]]
                                    # put the number of participant to label
                                    train_label_all_dic['Participant'] = [participant[0]]
                                    # put the number of video to label
                                    train_label_all_dic['Video'] = [video[0]]
                                else:
                                    train_label_all_dic[feature_name].append(episode[feature_name])
                                    # put the number of participant to label
                                    train_label_all_dic['Participant'].append(participant[0])
                                    # put the number of video to label
                                    train_label_all_dic['Video'].append(video[0])
                        elif participant[0] in test_P_list:
                            if ('bvp' in feature_name) or ('gsr' in feature_name) or ('skt' in feature_name):
                                if feature_name not in test_feature_all_dic.keys():
                                    test_feature_all_dic[feature_name] = [episode[feature_name]]
                                else:
                                    test_feature_all_dic[feature_name].append(episode[feature_name])
                            elif ('ReportadaBinarizado' in feature_name):
                                if feature_name not in test_label_all_dic.keys():
                                    test_label_all_dic[feature_name] = [episode[feature_name]]
                                    # put the number of participant to label
                                    test_label_all_dic['Participant'] = [participant[0]]
                                    # put the number of video to label
                                    test_label_all_dic['Video'] = [video[0]]
                                else:
                                    test_label_all_dic[feature_name].append(episode[feature_name])
                                    # put the number of participant to label
                                    test_label_all_dic['Participant'].append(participant[0])
                                    # put the number of video to label
                                    test_label_all_dic['Video'].append(video[0])

    # generate the ndarrar of features. For using feature-wise normalization
    train_feature_name_list = []
    train_feature_array = np.zeros(1)
    test_feature_name_list = []
    test_feature_array = np.zeros(1)
    for index, feature in enumerate(train_feature_all_dic.items()):
        if index == 0:
            train_feature_name_list.append(feature[0])
            train_feature_array = np.array(feature[1])
        else:
            train_feature_name_list.append(feature[0])
            train_feature_array = np.vstack((train_feature_array, np.array(feature[1])))
    for index, feature in enumerate(test_feature_all_dic.items()):
        if index == 0:
            test_feature_name_list.append(feature[0])
            test_feature_array = np.array(feature[1])
        else:
            test_feature_name_list.append(feature[0])
            test_feature_array = np.vstack((test_feature_array, np.array(feature[1])))
    # generate the label array.
    train_label_array = np.array(train_label_all_dic['ReportadaBinarizado'])
    # print('train list:' + str(train_P_list))
    # print('test list:' + str(test_P_list))
    test_label_array = np.array(test_label_all_dic['ReportadaBinarizado'])

    train_feature_array = train_feature_array.transpose()
    test_feature_array = test_feature_array.transpose()
    ## change nan to 0
    train_feature_array[np.isnan(train_feature_array)] = 0
    train_feature_array[np.isinf(train_feature_array)] = 0
    test_feature_array[np.isnan(test_feature_array)] = 0
    test_feature_array[np.isinf(test_feature_array)] = 0
    # get feature names
    train_P_label = train_label_all_dic['Participant']
    test_P_label = test_label_all_dic['Participant']
    # get feature names
    train_V_label = train_label_all_dic['Video']
    test_V_label = test_label_all_dic['Video']

    return train_feature_name_list, test_feature_name_list, train_feature_array, test_feature_array, \
        train_label_array, test_label_array, train_P_label, test_P_label, train_V_label, test_V_label

def data_extraction(json_file):
    train_feature_all_dic = {}
    train_label_all_dic = {}
    test_feature_all_dic = {}
    test_label_all_dic = {}
    with open(json_file, "r", encoding="utf-8") as f:
        content = json.load(f)
    ## Split train and test based on paticipants.
    ## Warning: data from one participant should not be included both in train and test
    train_P_list, test_P_list = get_train_test(content)

    ## Put all the data into one ndarray
    for participant in content.items():
        for video in participant[1].items():
            for episode in video[1].values():
                if 'ReportadaBinarizado' in episode.keys():
                    for feature_name in episode.keys():
                        # Judge if the feature from the participants from train or test
                        if participant[0] in train_P_list:
                            if ('bvp' in feature_name) or ('gsr' in feature_name) or ('skt' in feature_name):
                                if feature_name not in train_feature_all_dic.keys():
                                    train_feature_all_dic[feature_name] = [episode[feature_name]]
                                else:
                                    train_feature_all_dic[feature_name].append(episode[feature_name])
                            elif ('ReportadaBinarizado' in feature_name):
                                if feature_name not in train_label_all_dic.keys():
                                    train_label_all_dic[feature_name] = [episode[feature_name]]
                                    # put the number of participant to label
                                    train_label_all_dic['Participant'] = [participant[0]]
                                    # put the number of video to label
                                    train_label_all_dic['Video'] = [video[0]]
                                else:
                                    train_label_all_dic[feature_name].append(episode[feature_name])
                                    # put the number of participant to label
                                    train_label_all_dic['Participant'].append(participant[0])
                                    # put the number of video to label
                                    train_label_all_dic['Video'].append(video[0])
                        elif participant[0] in test_P_list:
                            if ('bvp' in feature_name) or ('gsr' in feature_name) or ('skt' in feature_name):
                                if feature_name not in test_feature_all_dic.keys():
                                    test_feature_all_dic[feature_name] = [episode[feature_name]]
                                else:
                                    test_feature_all_dic[feature_name].append(episode[feature_name])
                            elif ('ReportadaBinarizado' in feature_name):
                                if feature_name not in test_label_all_dic.keys():
                                    test_label_all_dic[feature_name] = [episode[feature_name]]
                                    # put the number of participant to label
                                    test_label_all_dic['Participant'] = [participant[0]]
                                    # put the number of video to label
                                    test_label_all_dic['Video'] = [video[0]]
                                else:
                                    test_label_all_dic[feature_name].append(episode[feature_name])
                                    # put the number of participant to label
                                    test_label_all_dic['Participant'].append(participant[0])
                                    # put the number of video to label
                                    test_label_all_dic['Video'].append(video[0])

    # generate the ndarrar of features. For using feature-wise normalization
    train_feature_name_list = []
    train_feature_array = np.zeros(1)
    test_feature_name_list = []
    test_feature_array = np.zeros(1)
    for index, feature in enumerate(train_feature_all_dic.items()):
        if index == 0:
            train_feature_name_list.append(feature[0])
            train_feature_array = np.array(feature[1])
        else:
            train_feature_name_list.append(feature[0])
            train_feature_array = np.vstack((train_feature_array, np.array(feature[1])))
    for index, feature in enumerate(test_feature_all_dic.items()):
        if index == 0:
            test_feature_name_list.append(feature[0])
            test_feature_array = np.array(feature[1])
        else:
            test_feature_name_list.append(feature[0])
            test_feature_array = np.vstack((test_feature_array, np.array(feature[1])))
    # generate the label array.
    train_label_array = np.array(train_label_all_dic['ReportadaBinarizado'])
    test_label_array = np.array(test_label_all_dic['ReportadaBinarizado'])

    train_feature_array = train_feature_array.transpose()
    test_feature_array = test_feature_array.transpose()
    ## change nan to 0
    train_feature_array[np.isnan(train_feature_array)] = 0
    train_feature_array[np.isinf(train_feature_array)] = 0
    test_feature_array[np.isnan(test_feature_array)] = 0
    test_feature_array[np.isinf(test_feature_array)] = 0
    # get feature names
    train_P_label = train_label_all_dic['Participant']
    test_P_label = test_label_all_dic['Participant']
    # get feature names
    train_V_label = train_label_all_dic['Video']
    test_V_label = test_label_all_dic['Video']

    return train_feature_name_list, test_feature_name_list, train_feature_array, test_feature_array, \
        train_label_array, test_label_array, train_P_label, test_P_label, train_V_label, test_V_label


if __name__ == '__main__':
    train_feature_name_list, test_feature_name_list, train_feature_array, test_feature_array, \
        train_label_array, test_label_array, train_P_label, test_P_label, \
        train_V_label, test_V_label = data_extraction('./Pack_all_data.json')
    train_filter_array = Abnormal_detection(train_feature_array)
    test_filter_array = Abnormal_detection(test_feature_array)
