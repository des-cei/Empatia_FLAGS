"""
                                                    ⠀⠀⠀⠀⠀⠀⠀⢠⣤⣀⠀⠀⠀⠀⢀⣀⣤⣤⠀⠀⠀⠀⠀⠀⠀
                                                    ⠀⠀⢀⢀⠀⠀⠀⢸⡿⠛⠛⠛⠛⠛⠉⠛⢿⣿⠀⠀⠀⠀⠀⠀⠀
 _   _       _     _____         _                    ⠀⠠⣿⣿⣿⣄⠀⣼⠀⠀⠉⣍⣀⣀⡍⠁⠀⢹⡀⠀⠀⠀⠀⠀⠀
| \ | | ___ | |_  |_   _|__   __| | __ _ _   _        ⠀⢸⣿⣿⣿⣿⡷⠋⠈⠀⠀⠀⠀⠀⠀⠀⠈⠘⠣⡀⠀⠀⠀⠀⠀
|  \| |/ _ \| __|   | |/ _ \ / _` |/ _` | | | |       ⠀⠈⣿⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣷⣦⡀⠀⠀
| |\  | (_) | |_    | | (_) | (_| | (_| | |_| |       ⠀⠀⢹⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⣿⣦⠀
|_| \_|\___/ \__|   |_|\___/ \__,_|\__,_|\__, |       ⠀⠀⣸⣿⣿⣶⣶⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇
                                         |___/        ⠀⣤⡟⠛⠋⠉⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠉⠈⠋⠈⢿⣿⡿
                                                    ⢀⡉⠀⠀⣀⣤⣄⢈⣿⣿⣿⣿⣿⣿⣿⣿⣿⢀⣤⣤⣄⠀⠀⣴⡄
                                                    ⠘⢇⠀⠰⣿⣿⢟⢼⣿⣿⣿⣿⣿⣿⣿⣿⡿⢜⠿⠿⠿⠀⡀⠀⠀
@Author: Junjiao Sun
@Time : Created in 1:25 PM 2024/10/11
@FileName: Train_Frozen_Classification_cluster.py
@Software: PyCharm


Introduction of this File:
This is the main file for training. We trained three gcn models at first, then freeze them and transfer to
a combined model for emotion recognition.
"""
import time
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
from thop import profile, clever_format # 新增导入 thop

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
                                                            train_cluster=now_test_trainVol, test_cluster=now_test_testVol,
                                                            approach_num=approach)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        print('Training GCN for BVP')

        gcn_epoch = 10
        # define the process of each signal
        def run_gcn_bvp():
            global gcn_bvp
            gcn_bvp = train_BVP(train_array_all, test_array_all, train_epoch=gcn_epoch)

        def run_gcn_gsr():
            global gcn_gsr
            gcn_gsr = train_GSR(train_array_all, test_array_all, train_epoch=gcn_epoch)

        def run_gcn_skt():
            global gcn_skt
            gcn_skt = train_SKT(train_array_all, test_array_all, train_epoch=gcn_epoch)

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

        print("All GCN models have finished training.")

        batch_size_all = 256

        # import all the data from three signals.
        train_data_group_bvp, test_data_group_bvp = BVP_tree(train_array_all, test_array_all)
        train_data_group_gsr, test_data_group_gsr = GSR_tree(train_array_all, test_array_all)
        train_data_group_skt, test_data_group_skt = SKT_tree(train_array_all, test_array_all)
        train_emotion_labels, test_emotion_labels = Get_emotion_label(train_array_all, test_array_all)

        train_loader_bvp = DataLoader(train_data_group_bvp, batch_size=batch_size_all, shuffle=True, drop_last=True)
        train_loader_gsr = DataLoader(train_data_group_gsr, batch_size=batch_size_all, shuffle=True, drop_last=True)
        train_loader_skt = DataLoader(train_data_group_skt, batch_size=batch_size_all, shuffle=True, drop_last=True)
        train_loader_label = DataLoader(train_emotion_labels, batch_size=batch_size_all, shuffle=True, drop_last=True)

        test_loader_bvp = DataLoader(test_data_group_bvp, batch_size=batch_size_all, shuffle=True, drop_last=True)
        test_loader_gsr = DataLoader(test_data_group_gsr, batch_size=batch_size_all, shuffle=True, drop_last=True)
        test_loader_skt = DataLoader(test_data_group_skt, batch_size=batch_size_all, shuffle=True, drop_last=True)
        test_loader_label = DataLoader(test_emotion_labels, batch_size=batch_size_all, shuffle=True, drop_last=True)

        epochs = 100
        combined_model = CombinedGCN_cnn_lstm(gcn_bvp, gcn_gsr, gcn_skt, out_channels=906, num_classes=2,
                                     batch_size=batch_size_all, lstm_hidden=141).to(device)
        optimizer = torch.optim.Adam(combined_model.fc.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

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

            for data1, data2, data3, labels in zip(train_loader_bvp, train_loader_gsr, train_loader_skt,
                                                   train_loader_label):
                optimizer.zero_grad()
                data_bvp = data1
                data_gsr = data2
                data_skt = data3
                label = labels.long().to(device)

                x_bvp, edge_index_bvp = data_bvp.x.to(device), data_bvp.edge_index.to(device)
                x_gsr, edge_index_gsr = data_gsr.x.to(device), data_gsr.edge_index.to(device)
                x_skt, edge_index_skt = data_skt.x.to(device), data_skt.edge_index.to(device)

                # forward
                out = combined_model(x_bvp, edge_index_bvp, x_gsr, edge_index_gsr, x_skt, edge_index_skt)

                # loss
                loss = criterion(out, label)

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
                    test_loss = loss

                    _, predicted = out.max(1)
                    numpy_value = predicted.cpu().numpy()
                    total_test += label.size(0)
                    correct_test += predicted.eq(label).sum().item()
                    prob_all_test.extend(predicted.cpu().numpy())
                    label_all_test.extend(label.cpu().numpy())

            # accuracy
            f1 = f1_score(label_all_test, prob_all_test, average='weighted')
            acc = accuracy_score(label_all_test, prob_all_test)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {test_loss:.4f}, "
                  f"Test Accuracy: {(100. * acc):.2f}, "
                  f"Test F1: {(100. * f1):.2f}")

            # ----------------------------------------------------
            # 新增：在最后一个 Epoch 结束后计算并输出 FLOPs 和 Params
            # ----------------------------------------------------
            if epoch == epochs - 1:
                try:
                    # 获取真实的单次 Batch Size
                    current_batch_size = x_bvp.shape[0] // 92

                    # 单独提取出三个 Encoders
                    encoder_bvp = combined_model.gcn1
                    encoder_gsr = combined_model.gcn2
                    encoder_skt = combined_model.gcn3

                    # 1. 使用原生 PyTorch 计算参数量 (100% 准确，无视底层算子限制)
                    params_bvp = sum(p.numel() for p in encoder_bvp.parameters() if p.requires_grad)
                    params_gsr = sum(p.numel() for p in encoder_gsr.parameters() if p.requires_grad)
                    params_skt = sum(p.numel() for p in encoder_skt.parameters() if p.requires_grad)
                    total_encoder_params = params_bvp + params_gsr + params_skt


                    # 2. 教 thop 如何计算 PyG GCNConv 的 FLOPs
                    def count_gcn(m, x, y):
                        # x[0] 是节点特征矩阵 [num_nodes, in_channels]
                        # x[1] 是 edge_index [2, num_edges]
                        x_feat = x[0]
                        num_nodes = x_feat.shape[0]

                        # (1) 线性映射的乘加次数: N * in_channels * out_channels
                        macs = num_nodes * m.in_channels * m.out_channels

                        # (2) 消息传递(Aggregation)的加法次数: E * out_channels
                        if len(x) > 1 and x[1] is not None:
                            num_edges = x[1].shape[1]
                            macs += num_edges * m.out_channels

                        m.total_ops += torch.DoubleTensor([int(macs)])


                    custom_ops = {GCNConv: count_gcn}

                    # 3. 计算 MACs (传入 custom_ops)
                    macs_bvp, _ = profile(encoder_bvp, inputs=(x_bvp, edge_index_bvp), custom_ops=custom_ops,
                                          verbose=False)
                    macs_gsr, _ = profile(encoder_gsr, inputs=(x_gsr, edge_index_gsr), custom_ops=custom_ops,
                                          verbose=False)
                    macs_skt, _ = profile(encoder_skt, inputs=(x_skt, edge_index_skt), custom_ops=custom_ops,
                                          verbose=False)

                    total_encoder_macs = macs_bvp + macs_gsr + macs_skt

                    # FLOPs 通常约等于 MACs 的 2 倍
                    encoder_flops = total_encoder_macs * 2.0
                    encoder_flops_per_sample = encoder_flops / current_batch_size

                    # 格式化输出
                    flops_str, params_str = clever_format([encoder_flops, total_encoder_params], "%.3f")
                    flops_per_sample_str, _ = clever_format([encoder_flops_per_sample, total_encoder_params], "%.3f")

                    print("\n" + "🔥" * 25)
                    print("📊 ENCODER ONLY Complexity Analysis (GCN - FLAGS)")
                    print("🔥" * 25)
                    print(f"Encoder Parameters : {params_str}")
                    print(f"Encoder FLOPs (BS={current_batch_size}) : {flops_str}")
                    print(f"Encoder FLOPs Per Sample : {flops_per_sample_str}")
                    print("=" * 50 + "\n")
                except Exception as e:
                    print(f"\n[Warning] FLOPs calculation failed: {e}")

        # save (取消注释并命名为 gcn_model.pth)
        torch.save(combined_model, "gcn_model.pth")