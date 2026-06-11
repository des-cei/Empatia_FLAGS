import os
import sys
sys.path.append("/home/junjiao/PycharmProjects/Empatia_IT06/")
sys.path.append("C:\\Users\\JUNJIAO SUN\\PycharmProjects\\Empatia")
import numpy as np

from Feature_wise_Normalization.Data_extraction import data_extraction
from Feature_wise_Normalization.Data_extraction import Abnormal_detection
import sklearn.datasets as dt
from FWN_main.normalization import DWN, FWN, plots
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import neighbors, svm
from FWN_main.methods import wrapper
import numpy

log_file = 'log_data_normalization_IT06_knn.log'


def Normalize0to1(array):
    '''
    Normalize the array
    '''
    mx = np.nanmax(array)
    mn = np.nanmin(array)
    # Suppress/hide the warning
    np.seterr(invalid='ignore')
    t = (array - mn) / (mx - mn)
    return t


def main_split(train_feature_name_list, train_filter_array,
               test_filter_array, train_label_array, test_label_array):
    X_train, X_test, y_train, y_test = train_filter_array, test_filter_array, train_label_array, test_label_array

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    clf = neighbors.KNeighborsClassifier(7)
    # clf = svm.SVC(kernel='rbf')
    d = DWN(methods='MS')
    f = FWN(methods='MS', Population=10, Iterations=100, cpus=None, viewi=0)
    # best, sol, conv, mapping = f.fit(X_test, y_test, clf, cv)
    best, sol, conv, mapping = f.fit(X_train, y_train, clf, cv)
    d_errs = d.fitcv(X_train, y_train, clf, cv)
    print('Cross-validation Errors:')
    for i, j in zip(mapping.values(), numpy.around(d_errs, decimals=3)):
        print(i, '-->', j)
    print('FWN', '-->', numpy.around(best, decimals=3))
    p = plots()
    p.make_plot(conv, d_errs)

    print('Prediction Errors:')
    best_p = f.predict(X_train, y_train, X_test, y_test, clf, sol, mapping)
    d_errs = d.fit(X_train, y_train, X_test, y_test, clf)
    for i, j in zip(mapping.values(), numpy.around(d_errs, decimals=3)):
        print(i, '-->', j)
    print('FWN', '-->', numpy.around(best_p, decimals=3))

    # Output the normalization functions with features name
    # print('=============Each feature with its normalization function=============')
    Normalization_list = []
    for index, nor_num in enumerate(sol):
        # print(str(feature_name_list[index]) + '-->' + str(mapping[int(nor_num)]) + '-->' + str(nor_num))
        Normalization_list.append(
            str(train_feature_name_list[index]) + '-->' + str(mapping[int(nor_num)]) + '-->' + str(nor_num))

    return best_p, Normalization_list


def Normalize_data(Normalization_log_file, feature_array, If0to1=False):
    # using the Normalization functions list, as well as the algorithms defined in FWN, to generate
    # the feature map after normalization.
    Xn_all = np.zeros(feature_array.shape[0])
    feature_array = feature_array.transpose()
    # Reload the file to extract the Normalization list.
    with open(Normalization_log_file, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if 'Over all Best N:' in line:
                Normalization_line = line.replace('Over all Best N: ---------------', '')
    # Normalization_line = Normalization_line.replace('[', '')
    # Normalization_line = Normalization_line.replace(']', '')
    # Normalization_line = Normalization_line.replace('\n', '')
    # Normalization_line = Normalization_line.replace("'", '')
    Normalization_list_all = Normalization_line.split(',')
    for index, Normalize_item in enumerate(Normalization_list_all):
        if '-->' in Normalize_item:
            Normalize_item_list = Normalize_item.split("-->")
            function = Normalize_item_list[1]
            wr = wrapper()
            x = feature_array[index]
            Xn, _ = wr.ndata_Empatia(data=feature_array[index], method=function)
            # If normalize again to 0 - 1
            if If0to1:
                Xn = Normalize0to1(Xn)
            if index == 0:
                Xn_all = Xn
            else:
                Xn_all = np.vstack((Xn_all, Xn))
    return Xn_all


if __name__ == '__main__':
    train_feature_name_list, test_feature_name_list, train_feature_array, test_feature_array, \
        train_label_array, test_label_array, train_P_label, test_P_label, \
        train_V_label, test_V_label = data_extraction('./Pack_all_data.json')
    print('load json finished')
    ## Use of not use the Abnormal detection
    train_feature_array = Abnormal_detection(train_feature_array)
    test_feature_array = Abnormal_detection(test_feature_array)
    ## Training 100 times to find a good way to normalize
    train_times = 100
    Best_Prediction = 1
    Normalization_list = []
    for i in range(train_times):
        now_P, now_N = main_split(train_feature_name_list, train_feature_array,
                                  test_feature_array, train_label_array, test_label_array)
        if now_P < Best_Prediction:
            Best_Prediction = now_P
            Normalization_list = now_N
            try:
                if not os.path.exists(log_file):
                    with open(log_file, "w", encoding='utf-8') as f:
                        f.write('Change to Best P:  ' + str(Best_Prediction) + ",\n")
                        print('Change to Best P:  ' + str(Best_Prediction))
                        f.write('Change to Best N:  ' + str(Normalization_list) + ",\n")
                        print('Change to Best N:  ' + str(Normalization_list))
                else:
                    with open(log_file, "a", encoding='utf-8') as f:
                        f.write('Change to Best P:  ' + str(Best_Prediction) + ",\n")
                        print('Change to Best P:  ' + str(Best_Prediction))
                        f.write('Change to Best N:  ' + str(Normalization_list) + ",\n")
                        print('Change to Best N:  ' + str(Normalization_list))
            except Exception as e:
                print("write error==>", e)
    try:
        if not os.path.exists(log_file):
            with open(log_file, "w", encoding='utf-8') as f:
                f.write('Over all Best P: ---------------   ' + str(Best_Prediction) + ",\n")
                print('Over all Best P: ---------------   ' + str(Best_Prediction))
                f.write('Over all Best N: ---------------' + str(Normalization_list) + ",\n")
                print('Over all Best N: ---------------' + str(Normalization_list))
        else:
            with open(log_file, "a", encoding='utf-8') as f:
                f.write('Over all Best P: ---------------   ' + str(Best_Prediction) + ",\n")
                print('Over all Best P: ---------------   ' + str(Best_Prediction))
                f.write('Over all Best N: ---------------' + str(Normalization_list) + ",\n")
                print('Over all Best N: ---------------' + str(Normalization_list))
    except Exception as e:
        print("write error==>", e)
