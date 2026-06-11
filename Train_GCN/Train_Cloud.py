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
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GCNConv
import threading
from Feature_wise_Normalization.Create_feature_maps import get_normalization
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


if __name__ == '__main__':
    log_file = '../Feature_wise_Normalization/log_data_normalization_IT06_knn.log'
    data_file = '../Feature_wise_Normalization/Pack_all_data.json'

    train_array_all, test_array_all = get_normalization(data_file, log_file)
    all_array_all = np.hstack((train_array_all, test_array_all))

    # use loso to separate a Vol
    Vol_list = list(map(int, set(all_array_all[124, :])))

    for Vol_test in Vol_list:
        print("Now for training Vol: " + str(Vol_test))
        mask = all_array_all[124, :] == Vol_test
        train_gcn_all = all_array_all[:, ~mask]
        test_Vol_data = all_array_all[:, mask]

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
        folder_name = os.path.join('GCN_models', 'GCN_model_Vol_' + str(int(Vol_test)))
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

        print('Saved GCNs for Vol: ' + str(int(Vol_test)))
