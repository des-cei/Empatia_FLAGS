import numpy as np
import matplotlib.pyplot as plt

from Feature_wise_Normalization.Data_extraction import data_extraction


def data_prepare(json_file='../Feature_wise_Normalization/Pack_all_data.json'):
    train_feature_name_list, test_feature_name_list, train_feature_array, test_feature_array, \
        train_label_array, test_label_array, train_P_label, test_P_label, \
        train_V_label, test_V_label = data_extraction(json_file)

    ## Get the positions of Abnormal data. For analyzing the reasons
    # Transfer the P_labels to array
    train_P_label = np.array([int(i.replace('V', '')) for i in train_P_label])
    test_P_label = np.array([int(i.replace('V', '')) for i in test_P_label])
    # Transfer the V_labels to array
    a = test_V_label[0]
    train_V_label = np.array([float(i) for i in train_V_label])
    test_V_label = np.array([float(i) for i in test_V_label])
    # Stack all the train and test. Include labels and P numbers.
    train_feature_array = train_feature_array.transpose()
    test_feature_array = test_feature_array.transpose()
    train_array_all = np.vstack((train_feature_array, train_label_array, train_P_label, train_V_label))
    test_array_all = np.vstack((test_feature_array, test_label_array, test_P_label, test_V_label))
    return train_array_all, test_array_all, test_feature_name_list


def Abnormal_detection_getPosition(input_array):
    # This function for filter the abnormal data
    feature_array = input_array[:123, :]
    all_abnormal_data = []
    for index_f, feature in enumerate(feature_array):
        Percentile = np.percentile(feature, [0, 25, 50, 75, 100])
        IQR = Percentile[3] - Percentile[1]
        UpLimit = Percentile[3] + IQR * 1.5
        DownLimit = Percentile[1] - IQR * 1.5
        # If the data more than Uplimit or less than Downlimit, see it as abnormal data
        # Use the average between former one and later one to replace it
        for index in range(len(feature) - 1):
            if (feature[index] > UpLimit) or (feature[index] < DownLimit):
                try:
                    print("Up and Down:")
                    print(UpLimit)
                    print(DownLimit)
                    print("feature:")
                    print(feature[index - 3:index + 2])
                except:
                    continue

                abnormal_dic = {}
                abnormal_feature_num = index_f
                abnormal_feature_position = input_array[123:, index]
                abnormal_dic['Feature_num'] = abnormal_feature_num
                abnormal_dic['Feature_position'] = abnormal_feature_position
                all_abnormal_data.append(abnormal_dic)
    return all_abnormal_data


def generate_features_plot(input_array, feature_name, set_type):
    # This function for filter the abnormal data
    feature_array = input_array[:123, :]
    label_array = input_array[123:, :]
    x_label_all = []
    flag = 0
    now_v = 0
    for i in range(len(label_array[0])):
        if now_v != int(label_array[2][i]):
            flag = 1
            now_v = int(label_array[2][i])
        else:
            flag += 1
        x_label = str(int(label_array[1][i])) + '-' + str(int(label_array[2][i])) + '-' + str(flag) + '-' + str(
            int(label_array[0][i]))
        x_label_all.append(x_label)
    all_abnormal_data = []
    for index_f, feature in enumerate(feature_array):
        Percentile = np.percentile(feature, [0, 25, 50, 75, 100])
        IQR = Percentile[3] - Percentile[1]
        UpLimit = Percentile[3] + IQR * 1.5
        DownLimit = Percentile[1] - IQR * 1.5
        # If the data more than Uplimit or less than Downlimit, see it as abnormal data
        # Use the average between former one and later one to replace it
        for index in range(len(feature) - 1):
            if (feature[index] > UpLimit) or (feature[index] < DownLimit):
                abnormal_dic = {}
                abnormal_feature_num = index_f
                abnormal_feature_position = input_array[123:, index]
                abnormal_dic['Feature_num'] = abnormal_feature_num
                abnormal_dic['Feature_position'] = abnormal_feature_position
                all_abnormal_data.append(abnormal_dic)
        ## Draw plot of all date
        # plt.figure(dpi=50).set_figheight(15)
        # Distinguish the fear and no-fear
        colors = []
        for index, value in enumerate(x_label_all[::3]):
            value_list = value.split('-')
            if value_list[3] == '1':
                colors.append('orange')
            else:
                colors.append('blue')

        plt.figure(figsize=(512, 10), dpi=60)
        plt.scatter(x_label_all[::3], feature[::3], s=200, c=colors)
        plt.axhline(UpLimit, color='red')
        plt.axhline(DownLimit, color='red')
        plt.xticks(rotation=90)
        if set_type == 'test':
            plt.savefig('./test_features/' + feature_name[index_f] + '.jpg')
        if set_type == 'train':
            plt.savefig('./train_features/' + feature_name[index_f] + '.jpg')
        # plt.show()
        print(feature_name[index_f])
        print('Done')
        plt.close()


if __name__ == '__main__':
    train_array_all, test_array_all, test_feature_name_list = data_prepare(
        json_file='../Feature_wise_Normalization/Pack_all_data.json')
    # train_filter_array = Abnormal_detection_getPosition(train_array_all)
    # test_filter_array = Abnormal_detection_getPosition(test_array_all)
    # generate_features_plot(test_array_all, test_feature_name_list, set_type='test')
    generate_features_plot(train_array_all, test_feature_name_list, set_type='train')
