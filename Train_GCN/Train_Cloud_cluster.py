"""
                                                    ⠀⠀⠀⠀⠀⠀⠀⢠⣤⣀⠀⠀⠀⠀⢀⣀⣤⣤⠀⠀⠀⠀⠀⠀⠀
                                                    ⠀⠀⢀⢀⠀⠀⠀⢸⡿⠛⠛⠛⠛⠛⠉⠛⢿⣿⠀⠀⠀⠀⠀⠀⠀
 _   _       _     _____         _             		⠀⠠⣿⣿⣿⣄⠀⣼⠀⠀⠉⣍⣀⣀⡍⠁⠀⢹⡀⠀⠀⠀⠀⠀⠀
| \ | | ___ | |_  |_   _|__   __| | __ _ _   _ 		⠀⢸⣿⣿⣿⣿⡷⠋⠈⠀⠀⠀⠀⠀⠀⠀⠈⠘⠣⡀⠀⠀⠀⠀⠀
|  \| |/ _ \| __|   | |/ _ \ / _` |/ _` | | | |		⠀⠈⣿⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣷⣦⡀⠀⠀
| |\  | (_) | |_    | | (_) | (_| | (_| | |_| |		⠀⠀⢹⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⣿⣦⠀
|_| \_|\___/ \__|   |_|\___/ \__,_|\__,_|\__, |		⠀⠀⣸⣿⣿⣶⣶⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇
                                         |___/ 		⠀⣤⡟⠛⠋⠉⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠉⠈⠋⠈⢿⣿⡿
                                                    ⢀⡉⠀⠀⣀⣤⣄⢈⣿⣿⣿⣿⣿⣿⣿⣿⣿⢀⣤⣤⣄⠀⠀⣴⡄
                                                    ⠘⢇⠀⠰⣿⣿⢟⢼⣿⣿⣿⣿⣿⣿⣿⣿⡿⢜⠿⠿⠿⠀⡀⠀⠀
@Author: Junjiao Sun                               
@Time : Created in 10:50 AM 2025/03/11   
@FileName: Train_Cloud.py                           
@Software: PyCharm


Introduction of this File:

"""
import os
import time
from random import shuffle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GCNConv
import threading
from Feature_wise_Normalization.Create_feature_maps import get_normalization, get_normalization_cluster
from Train_GCN.CombinedGCN_cnn import CombinedGCN_cnn
from Train_GCN.CombinedGCN_cnn_lstm import CombinedGCN_cnn_lstm
from Train_GCN.Generate_tree import BVP_tree, SKT_tree, GSR_tree, Get_emotion_label
from Train_GCN.Train_BVP_GCN import train_BVP
from Train_GCN.Train_GSR_GCN import train_GSR
from Train_GCN.Train_SKT_GCN import train_SKT
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import accuracy_score, f1_score


def get_train_test_gcn(all_array):
    Participants_list = list(set(all_array[-2, :]))
    # Randomly split participants
    n_total = len(Participants_list)
    offset = round(n_total * 0.8)
    if n_total == 0 or offset < 1:
        return [], Participants_list
    # random split the train and test
    shuffle(Participants_list)
    train_P_list = Participants_list[:offset]
    test_P_list = Participants_list[offset:]

    mask_train = np.isin(all_array[-2, :], train_P_list)
    indices_train = np.where(mask_train)[0]
    mask_test = np.isin(all_array[-2, :], test_P_list)
    indices_test = np.where(mask_test)[0]

    array_train = all_array[:, indices_train]
    array_test = all_array[:, indices_test]

    return array_train, array_test


def get_train_test_classifier(all_array):
    array_train = all_array[:, :int(all_array.shape[1] * 0.2)]
    array_test = all_array[:, int(all_array.shape[1] * 0.2):]

    return array_train, array_test

def get_clusters_loso(csv_path, test_vol):
    df = pd.read_csv(csv_path)
    column_cluster = df['id_cluster']
    column_method = df['method']
    column_subject = df['id_vol']
    column_test = df['vol_test']
    # get the iteration for loso
    column_test_index = np.where(column_test == test_vol)
    cluster_now = column_cluster[column_test_index[0][-1]]
    column_method_M2_train = np.where(column_method == 'method2_train')
    column_method_M2_test = np.where(column_method == 'method2_test')
    column_cluster_index = np.where(column_cluster == cluster_now)
    # find the intersection of test vol and method2 train.
    now_test_trainIndex = np.intersect1d(np.intersect1d(column_test_index[0], column_method_M2_train[0]),
                                         column_cluster_index[0])
    now_test_trainVol = column_subject[now_test_trainIndex].values
    now_test_testVol = [test_vol]
    return now_test_trainVol, now_test_testVol

if __name__ == '__main__':
    log_file = '../Feature_wise_Normalization/log_data_normalization_IT06_knn.log'
    data_file = '../Feature_wise_Normalization/Pack_all_data.json'
    approach = 'approach1'

    excel_file = '../clustering_WEMAC_123feat_v03feb.csv'

    df = pd.read_csv(excel_file)
    column_cluster = df['id_cluster']
    column_method = df['method']
    column_subject = df['id_vol']
    column_test = df['vol_test']
    test_set = set(column_test)

    for test_index, test_vol in enumerate(test_set):
        print('Now for training Vol: ' + str(int(test_vol)))
        # Extract the clusters from .csv
        now_test_trainVol, now_test_testVol = get_clusters_loso(excel_file, int(test_vol))

        train_array_all, test_array_all = get_normalization_cluster(data_file, log_file,
                                                                    train_cluster=now_test_trainVol,
                                                                    test_cluster=now_test_testVol,
                                                                    approach_num=approach)
        all_array_all = np.hstack((train_array_all, test_array_all))

        mask = all_array_all[124, :] == test_vol
        train_gcn_all = all_array_all[:, ~mask]

        train_gcn_train, train_gcn_test = get_train_test_gcn(train_gcn_all)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        gcn_epoch = 10


        # define the process of each signal
        def run_gcn_bvp():
            global gcn_bvp
            gcn_bvp = train_BVP(train_gcn_train, train_gcn_test, train_epoch=gcn_epoch)


        def run_gcn_gsr():
            global gcn_gsr
            gcn_gsr = train_GSR(train_gcn_train, train_gcn_test, train_epoch=gcn_epoch)


        def run_gcn_skt():
            global gcn_skt
            gcn_skt = train_SKT(train_gcn_train, train_gcn_test, train_epoch=gcn_epoch)


        # create three Thread
        thread_bvp = threading.Thread(target=run_gcn_bvp)
        thread_gsr = threading.Thread(target=run_gcn_gsr)
        thread_skt = threading.Thread(target=run_gcn_skt)

        thread_bvp.start()
        thread_gsr.start()
        thread_skt.start()

        thread_bvp.join()
        thread_gsr.join()
        thread_skt.join()

        # Save all the trained GCNs
        # Create the folder for the gcn models
        folder_name = os.path.join('GCN_models', 'GCN_model_Vol_' + str(int(test_vol)))
        try:
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created successfully!")
        except FileExistsError:
            print(f"Folder '{folder_name}' already exists.")
        bvp_path = os.path.join(folder_name, 'gcn_bvp.pth')
        torch.save(gcn_bvp, bvp_path)

        gsr_path = os.path.join(folder_name, 'gcn_gsr.pth')
        torch.save(gcn_gsr, gsr_path)

        skt_path = os.path.join(folder_name, 'gcn_skt.pth')
        torch.save(gcn_skt, skt_path)

        print('Saved GCNs for Vol: ' + str(int(test_vol)))
