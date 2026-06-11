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
@Time : Created in 10:56 AM 2025/03/11   
@FileName: Train_Edge.py                           
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

def get_train_test_classifier(all_array):
    array_train = all_array[:, :int(all_array.shape[1] * 0.2)]
    array_test = all_array[:, int(all_array.shape[1] * 0.2):]

    return array_train, array_test


if __name__ == '__main__':
    log_file = '../Feature_wise_Normalization/log_data_normalization_IT06_knn.log'
    data_file = '../Feature_wise_Normalization/Pack_all_data.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_array_all, test_array_all = get_normalization(data_file, log_file)
    all_array_all = np.hstack((train_array_all, test_array_all))

    # use loso to separate a Vol
    Vol_list = list(map(int, set(all_array_all[124, :])))
    Root = 'GCN_models'

    for Vol_test in Vol_list:
        print("Now for training Vol: " + str(Vol_test))
        Vol_best_acc = 0
        Vol_best_f1 = 0
        # Load the gcn models for this vol
        Model_path = os.path.join(Root, 'GCN_model_Vol_' + str(int(Vol_test)))
        gcn_bvp = torch.load(os.path.join(Model_path, 'gcn_bvp.pth'))
        gcn_gsr = torch.load(os.path.join(Model_path, 'gcn_gsr.pth'))
        gcn_skt = torch.load(os.path.join(Model_path, 'gcn_skt.pth'))

        mask = all_array_all[124, :] == Vol_test
        train_gcn_all = all_array_all[:, ~mask]
        test_Vol_data = all_array_all[:, mask]
        test_Vol_train, test_Vol_test = get_train_test_classifier(test_Vol_data)

        batch_size_all = 16

        # gcn_bvp = train_BVP(train_array_all, test_array_all, train_epoch=3)
        # gcn_gsr = train_GSR(train_array_all, test_array_all, train_epoch=3)
        # gcn_skt = train_SKT(train_array_all, test_array_all, train_epoch=3)

        # import all the data from three signals.
        train_data_group_bvp, test_data_group_bvp = BVP_tree(test_Vol_train, test_Vol_test)
        train_data_group_gsr, test_data_group_gsr = GSR_tree(test_Vol_train, test_Vol_test)
        train_data_group_skt, test_data_group_skt = SKT_tree(test_Vol_train, test_Vol_test)
        train_emotion_labels, test_emotion_labels = Get_emotion_label(test_Vol_train, test_Vol_test)

        # pack all the data group
        # train_dataset_bvp, test_dataset_bvp = Pack_data_group(train_data_group_bvp, test_data_group_bvp)
        # train_dataset_gsr, test_dataset_gsr = Pack_data_group(train_data_group_gsr, test_data_group_gsr)
        # train_dataset_skt, test_dataset_skt = Pack_data_group(train_data_group_skt, test_data_group_skt)

        train_loader_bvp = DataLoader(train_data_group_bvp, batch_size=batch_size_all, shuffle=True, drop_last=True)
        train_loader_gsr = DataLoader(train_data_group_gsr, batch_size=batch_size_all, shuffle=True, drop_last=True)
        train_loader_skt = DataLoader(train_data_group_skt, batch_size=batch_size_all, shuffle=True, drop_last=True)
        train_loader_label = DataLoader(train_emotion_labels, batch_size=batch_size_all, shuffle=True, drop_last=True)

        test_loader_bvp = DataLoader(test_data_group_bvp, batch_size=batch_size_all, shuffle=True, drop_last=True)
        test_loader_gsr = DataLoader(test_data_group_gsr, batch_size=batch_size_all, shuffle=True, drop_last=True)
        test_loader_skt = DataLoader(test_data_group_skt, batch_size=batch_size_all, shuffle=True, drop_last=True)
        test_loader_label = DataLoader(test_emotion_labels, batch_size=batch_size_all, shuffle=True, drop_last=True)

        epochs = 100
        # combined_model = CombinedGCN(gcn_bvp, gcn_gsr, gcn_skt, out_channels=906, num_classes=2,
        #                              batch_size=batch_size_all).to(device)
        combined_model = CombinedGCN_cnn(gcn_bvp, gcn_gsr, gcn_skt, out_channels=906, num_classes=2,
                                         batch_size=batch_size_all).to(device)
        # combined_model = CombinedGCN_cnn_lstm(gcn_bvp, gcn_gsr, gcn_skt, out_channels=906, num_classes=2,
        #                              batch_size=batch_size_all, lstm_hidden=141).to(device)
        optimizer = torch.optim.Adam(combined_model.fc.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.BCELoss()
        print('Start training classification')
        for epoch in range(epochs):
            # train
            combined_model.train()
            train_loss = 0
            all_preds, all_labels = [], []
            correct_train = 0
            total_train = 0
            correct_test = 0
            total_test = 0
            label_all = []
            prob_all = []
            # for index, data in enumerate(train_data_group_bvp):
            for data1, data2, data3, labels in zip(train_loader_bvp, train_loader_gsr, train_loader_skt,
                                                   train_loader_label):
                optimizer.zero_grad()
                # get the input of three signals
                data_bvp = data1
                data_gsr = data2
                data_skt = data3
                label = labels.long().to(device)

                # data_bvp = train_data_group_bvp[index].to(device)
                # data_gsr = train_data_group_gsr[index].to(device)
                # data_skt = train_data_group_skt[index].to(device)
                # label = torch.tensor(int(train_emotion_labels[index])).to(device)

                x_bvp, edge_index_bvp = data_bvp.x.to(device), data_bvp.edge_index.to(device)
                x_gsr, edge_index_gsr = data_gsr.x.to(device), data_gsr.edge_index.to(device)
                x_skt, edge_index_skt = data_skt.x.to(device), data_skt.edge_index.to(device)

                # x_bvp, edge_index_bvp = data_bvp[0].to(device), data_bvp[1].to(device)
                # x_gsr, edge_index_gsr = data_gsr[0].to(device), data_gsr[1].to(device)
                # x_skt, edge_index_skt = data_skt[0].to(device), data_skt[1].to(device)
                # forward
                start_time = time.time()
                out = combined_model(x_bvp, edge_index_bvp, x_gsr, edge_index_gsr, x_skt, edge_index_skt)
                end_time = time.time()
                time_consumption = (end_time - start_time) * 1000
                print('time consumption for training one epoch (ms):' + str(time_consumption))

                # loss
                loss = criterion(out, label)
                # loss = criterion(out.squeeze(), label.float())

                # backward
                loss.backward()
                optimizer.step()
                train_loss = loss

                _, predicted = out.max(1)
                numpy_value = predicted.cpu().numpy()
                total_train += label.size(0)
                correct_train += predicted.eq(label).sum().item()
                prob_all.extend(predicted.cpu().numpy())
                label_all.extend(label.cpu().numpy())
                # print('batch: ' + str(index))

            # accuracy
            f1 = f1_score(label_all, prob_all, average='weighted')
            acc = accuracy_score(label_all, prob_all)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, "
                  f"Train Accuracy: {(100. * acc):.2f}, "
                  f"Train F1: {(100. * f1):.2f}")

            # test
            combined_model.eval()
            test_loss = 0

            label_all_test = []
            prob_all_test = []

            with torch.no_grad():
                for data1, data2, data3, labels in zip(test_loader_bvp, test_loader_gsr, test_loader_skt,
                                                       test_loader_label):
                    optimizer.zero_grad()
                    # get the input of three signals
                    data_bvp = data1
                    data_gsr = data2
                    data_skt = data3
                    label = labels.long().to(device)

                    x_bvp, edge_index_bvp = data_bvp.x.to(device), data_bvp.edge_index.to(device)
                    x_gsr, edge_index_gsr = data_gsr.x.to(device), data_gsr.edge_index.to(device)
                    x_skt, edge_index_skt = data_skt.x.to(device), data_skt.edge_index.to(device)

                    # forward
                    start_time = time.time()
                    out = combined_model(x_bvp, edge_index_bvp, x_gsr, edge_index_gsr, x_skt, edge_index_skt)
                    end_time = time.time()
                    time_consumption = (end_time - start_time) * 1000
                    print('time consumption for 1 time window (ms):' + str(time_consumption))

                    loss = criterion(out, label)
                    # loss = criterion(out.squeeze(), label.float())
                    test_loss = loss

                    #
                    _, predicted = out.max(1)
                    numpy_value = predicted.cpu().numpy()
                    total_test += label.size(0)
                    correct_test += predicted.eq(label).sum().item()
                    prob_all_test.extend(predicted.cpu().numpy())
                    label_all_test.extend(label.cpu().numpy())
                    # print('batch: ' + str(index))

            # accuracy
            f1 = f1_score(label_all_test, prob_all_test, average='weighted')
            acc = accuracy_score(label_all_test, prob_all_test)
            if acc > Vol_best_acc:
                Vol_best_acc = acc
            if f1 > Vol_best_f1:
                Vol_best_f1 = f1
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {test_loss:.4f}, "
                  f"Test Accuracy: {(100. * acc):.2f}, "
                  f"Test F1: {(100. * f1):.2f}")
            # # save
            # torch.save(combined_model, "model.pth")
        print('For Vol Num: ' + str(int(Vol_test)))
        print('best acc: ' + str(Vol_best_acc))
        print('best f1: ' + str(Vol_best_f1))
