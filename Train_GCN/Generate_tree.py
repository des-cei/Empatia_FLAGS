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
@Time : Created in 12:16 PM 2024/10/10   
@FileName: Generate_tree.py                           
@Software: PyCharm


Introduction of this File:
Three signals BVP, GSR and SKT. We designed 123 bio-feature for them. This file is to generate three tree for each
signal.
"""

import csv
import itertools
import sys
import os
import time

sys.path.append("/home/junjiao/PycharmProjects/Empatia_GCN/")
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import argparse
from Feature_wise_Normalization.Create_feature_maps import get_normalization, generate_feature_maps
import pandas as pd
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

def BVP_tree(train_array_all, test_array_all):
    print('Generate BVP tree')

    train_array_all = np.transpose(train_array_all)
    test_array_all = np.transpose(test_array_all)
    Test_Tree_group = []
    Train_Tree_group = []
    num_nodes_per_tree = 1 + 7 + 84
    for test_index, frame in enumerate(test_array_all):
        # Extract BVP
        BVP_array = frame[:84]
        BVP_PP_array = BVP_array[:14]
        BVP_HR_array = BVP_array[14:21]
        BVP_PW_array = BVP_array[21:35]
        BVP_PDT_array = BVP_array[35:49]
        BVP_PRT_array = BVP_array[49:63]
        BVP_PA_array = BVP_array[63:70]
        BVP_PWR_array = BVP_array[70:]
        # Generate the tree.
        # Embedding the non-leaf-node as int. BVP=0. PP=1. HR=2. PW=3. PDT=4. PRT=5. PA=6. PWR=7.
        edges = []
        labels = []
        for i in range(1, 8):
            edges.append([0, i])
        for i in range(8, 22):
            edges.append([1, i])
            labels.append(0)
        for i in range(22, 29):
            edges.append([2, i])
            labels.append(1)
        for i in range(29, 43):
            edges.append([3, i])
            labels.append(2)
        for i in range(43, 57):
            edges.append([4, i])
            labels.append(3)
        for i in range(57, 71):
            edges.append([5, i])
            labels.append(4)
        for i in range(71, 78):
            edges.append([6, i])
            labels.append(5)
        for i in range(78, 92):
            edges.append([7, i])
            labels.append(6)
        test_edge_index = torch.tensor(edges, dtype=torch.long).t()
        leaf_nodes = torch.tensor(BVP_array, dtype=torch.float)
        non_leaf_node = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
        test_nodes = torch.cat([non_leaf_node, leaf_nodes], dim=0)
        test_nodes_reshaped = test_nodes.unsqueeze(1)
        labels = torch.tensor(labels)
        leaf_node_idx = torch.arange(8, num_nodes_per_tree)

        tree_data = Data(x=test_nodes_reshaped, edge_index=test_edge_index, y=labels, train_mask=leaf_node_idx)
        Test_Tree_group.append(tree_data)

    for train_index, frame in enumerate(train_array_all):
        # Extract BVP
        BVP_array = frame[:84]
        BVP_PP_array = BVP_array[:14]
        BVP_HR_array = BVP_array[14:21]
        BVP_PW_array = BVP_array[21:35]
        BVP_PDT_array = BVP_array[35:49]
        BVP_PRT_array = BVP_array[49:63]
        BVP_PA_array = BVP_array[63:70]
        BVP_PWR_array = BVP_array[70:]
        # Generate the tree.
        # Embedding the non-leaf-node as int. BVP=0. PP=1. HR=2. PW=3. PDT=4. PRT=5. PA=6. PWR=7.
        edges = []
        labels = []
        for i in range(1, 8):
            edges.append([0, i])
        for i in range(8, 22):
            edges.append([1, i])
            labels.append(0)
        for i in range(22, 29):
            edges.append([2, i])
            labels.append(1)
        for i in range(29, 43):
            edges.append([3, i])
            labels.append(2)
        for i in range(43, 57):
            edges.append([4, i])
            labels.append(3)
        for i in range(57, 71):
            edges.append([5, i])
            labels.append(4)
        for i in range(71, 78):
            edges.append([6, i])
            labels.append(5)
        for i in range(78, 92):
            edges.append([7, i])
            labels.append(6)
        train_edge_index = torch.tensor(edges, dtype=torch.long).t()
        leaf_nodes = torch.tensor(BVP_array, dtype=torch.float)
        non_leaf_node = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
        train_nodes = torch.cat([non_leaf_node, leaf_nodes], dim=0)
        train_nodes_reshaped = train_nodes.unsqueeze(1)
        labels = torch.tensor(labels)
        leaf_node_idx = torch.arange(8, num_nodes_per_tree)

        tree_data = Data(x=train_nodes_reshaped, edge_index=train_edge_index, y=labels, train_mask=leaf_node_idx)
        Train_Tree_group.append(tree_data)

    return Train_Tree_group, Test_Tree_group

def GSR_tree(train_array_all, test_array_all):
    print('Generate GSR tree')
    train_array_all = np.transpose(train_array_all)
    test_array_all = np.transpose(test_array_all)
    Test_Tree_group = []
    Train_Tree_group = []
    num_nodes_per_tree = 1 + 6 + 34
    for test_index, frame in enumerate(test_array_all):
        # Extract GSR
        GSR_array = frame[84:118]
        GSR_signal_array = GSR_array[:2]
        GSR_PSD_array = GSR_array[2:12]
        GSR_peak_array = GSR_array[12:18]
        GSR_FD_array = GSR_array[18:24]
        GSR_SCL_array = GSR_array[24:28]
        GSR_SCR_array = GSR_array[28:]
        # Generate the tree.
        # Embedding the non-leaf-node as int. GSR=0. signal=1. PSD=2. peak=3. FD=4. SCL=5. SCR=6.
        edges = []
        labels = []
        for i in range(1, 7):
            edges.append([0, i])
        for i in range(7, 9):
            edges.append([1, i])
            labels.append(0)
        for i in range(9, 19):
            edges.append([2, i])
            labels.append(1)
        for i in range(19, 25):
            edges.append([3, i])
            labels.append(2)
        for i in range(25, 31):
            edges.append([4, i])
            labels.append(3)
        for i in range(31, 35):
            edges.append([5, i])
            labels.append(4)
        for i in range(35, 41):
            edges.append([6, i])
            labels.append(5)
        test_edge_index = torch.tensor(edges, dtype=torch.long).t()
        leaf_nodes = torch.tensor(GSR_array, dtype=torch.float)
        non_leaf_node = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)
        test_nodes = torch.cat([non_leaf_node, leaf_nodes], dim=0)
        test_nodes_reshaped = test_nodes.unsqueeze(1)
        labels = torch.tensor(labels)
        leaf_node_idx = torch.arange(7, num_nodes_per_tree)

        tree_data = Data(x=test_nodes_reshaped, edge_index=test_edge_index, y=labels, train_mask=leaf_node_idx)
        Test_Tree_group.append(tree_data)
    for train_index, frame in enumerate(train_array_all):
        # Extract GSR
        GSR_array = frame[84:118]
        GSR_signal_array = GSR_array[:2]
        GSR_PSD_array = GSR_array[2:12]
        GSR_peak_array = GSR_array[12:18]
        GSR_FD_array = GSR_array[18:24]
        GSR_SCL_array = GSR_array[24:28]
        GSR_SCR_array = GSR_array[28:]
        # Generate the tree.
        # Embedding the non-leaf-node as int. GSR=0. signal=1. PSD=2. peak=3. FD=4. SCL=5. SCR=6.
        edges = []
        labels = []
        for i in range(1, 7):
            edges.append([0, i])
        for i in range(7, 9):
            edges.append([1, i])
            labels.append(0)
        for i in range(9, 19):
            edges.append([2, i])
            labels.append(1)
        for i in range(19, 25):
            edges.append([3, i])
            labels.append(2)
        for i in range(25, 31):
            edges.append([4, i])
            labels.append(3)
        for i in range(31, 35):
            edges.append([5, i])
            labels.append(4)
        for i in range(35, 41):
            edges.append([6, i])
            labels.append(5)
        train_edge_index = torch.tensor(edges, dtype=torch.long).t()
        leaf_nodes = torch.tensor(GSR_array, dtype=torch.float)
        non_leaf_node = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)
        train_nodes = torch.cat([non_leaf_node, leaf_nodes], dim=0)
        train_nodes_reshaped = train_nodes.unsqueeze(1)
        labels = torch.tensor(labels)
        leaf_node_idx = torch.arange(7, num_nodes_per_tree)

        tree_data = Data(x=train_nodes_reshaped, edge_index=train_edge_index, y=labels, train_mask=leaf_node_idx)
        Train_Tree_group.append(tree_data)
    return Train_Tree_group, Test_Tree_group

def SKT_tree(train_array_all, test_array_all):
    print('Generate SKT tree')
    train_array_all = np.transpose(train_array_all)
    test_array_all = np.transpose(test_array_all)
    Test_Tree_group = []
    Train_Tree_group = []
    num_nodes_per_tree = 1 + 2 + 5
    for test_index, frame in enumerate(test_array_all):
        # Extract SKT
        SKT_array = frame[118:123]
        SKT_general = SKT_array[:3]
        SKT_temp = SKT_array[3:]
        # Generate the tree.
        # Embedding the non-leaf-node as int. SKT=0. general=1. temp=2.
        edges = []
        labels = []
        for i in range(1, 3):
            edges.append([0, i])
        for i in range(3, 6):
            edges.append([1, i])
            labels.append(0)
        for i in range(6, 8):
            edges.append([2, i])
            labels.append(1)
        test_edge_index = torch.tensor(edges, dtype=torch.long).t()
        leaf_nodes = torch.tensor(SKT_array, dtype=torch.float)
        non_leaf_node = torch.tensor([0, 1, 2], dtype=torch.long)
        test_nodes = torch.cat([non_leaf_node, leaf_nodes], dim=0)
        test_nodes_reshaped = test_nodes.unsqueeze(1)
        labels = torch.tensor(labels)
        leaf_node_idx = torch.arange(3, num_nodes_per_tree)

        tree_data = Data(x=test_nodes_reshaped, edge_index=test_edge_index, y=labels, train_mask=leaf_node_idx)
        Test_Tree_group.append(tree_data)
    for train_index, frame in enumerate(train_array_all):
        # Extract SKT
        SKT_array = frame[118:123]
        SKT_general = SKT_array[:3]
        SKT_temp = SKT_array[3:]
        # Generate the tree.
        # Embedding the non-leaf-node as int. SKT=0. general=1. temp=2.
        edges = []
        labels = []
        for i in range(1, 3):
            edges.append([0, i])
        for i in range(3, 6):
            edges.append([1, i])
            labels.append(0)
        for i in range(6, 8):
            edges.append([2, i])
            labels.append(1)
        train_edge_index = torch.tensor(edges, dtype=torch.long).t()
        leaf_nodes = torch.tensor(SKT_array, dtype=torch.float)
        non_leaf_node = torch.tensor([0, 1, 2], dtype=torch.long)
        train_nodes = torch.cat([non_leaf_node, leaf_nodes], dim=0)
        train_nodes_reshaped = train_nodes.unsqueeze(1)
        labels = torch.tensor(labels)
        leaf_node_idx = torch.arange(3, num_nodes_per_tree)

        tree_data = Data(x=train_nodes_reshaped, edge_index=train_edge_index, y=labels, train_mask=leaf_node_idx)
        Train_Tree_group.append(tree_data)
    return Train_Tree_group, Test_Tree_group

def Get_emotion_label(train_array_all, test_array_all):
    print('Get emotion labels')
    train_label = train_array_all[123, :]
    test_label = test_array_all[123, :]
    return train_label, test_label

if __name__ == '__main__':
    log_file = '../Feature_wise_Normalization/log_data_normalization_IT06_knn.log'
    data_file = '../Feature_wise_Normalization/Pack_all_data.json'

    train_array_all, test_array_all = get_normalization(data_file, log_file)

    train_data_group_bvp, test_data_group_bvp = BVP_tree(train_array_all, test_array_all)
    # train_data_group_gsr, test_data_group_gsr = GSR_tree(train_array_all, test_array_all)
    # train_data_group_skt, test_data_group_skt = SKT_tree(train_array_all, test_array_all)
    Get_emotion_label(train_array_all, test_array_all)
    print(1)
