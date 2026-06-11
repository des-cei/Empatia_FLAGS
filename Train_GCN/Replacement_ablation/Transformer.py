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
@Time : Created in 12:28 PM 2026/03/10
@FileName: MLP.py
@Software: PyCharm


Introduction of this File:

"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time
import torch
from Feature_wise_Normalization.Create_feature_maps import get_normalization, get_normalization_cluster
from Train_GCN.CombinedGCN_cnn_lstm import CombinedGCN_cnn_lstm
from Train_GCN.Generate_tree import BVP_tree, SKT_tree, GSR_tree, Get_emotion_label
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from thop import profile, clever_format # 新增导入 thop


class NodeTransformer(torch.nn.Module):
    def __init__(self, in_channels, d_model, out_channels, num_nodes, num_heads=4, num_layers=2):
        super(NodeTransformer, self).__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model

        # 1. Input projection layer: maps in_channels (e.g., 1D) to the Transformer hidden dimension d_model (e.g., 32)
        self.input_proj = nn.Linear(in_channels, d_model)

        # 2. Learnable Positional Encoding
        # Informs the Transformer of the relative positions of the 92/41/8 physiological features within the array
        self.pos_embed = nn.Parameter(torch.randn(1, num_nodes, d_model))

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Output projection layer: maps d_model back to the required downstream dimensions (e.g., 7, 6, 2)
        self.output_proj = nn.Linear(d_model, out_channels)

    def forward(self, x, edge_index=None):
        # Intentionally accepts edge_index but strictly ignores it to ensure a seamless drop-in replacement!

        # 1. Dynamically infer Batch Size
        B = x.shape[0] // self.num_nodes

        # 2. Convert to standard Transformer sequence format: (Batch, Sequence_Length, Features)
        # The shape here is (Batch, num_nodes, in_channels)
        x = x.view(B, self.num_nodes, -1)

        # 3. Project to d_model dimension and add positional encoding
        x = self.input_proj(x)
        x = x + self.pos_embed  # Broadcasting (1, num_nodes, d_model) to (Batch, num_nodes, d_model)

        # 4. Pass through the self-attention layers
        x = self.transformer_encoder(x)

        # 5. Project to the target output dimension out_channels
        x = self.output_proj(x)

        # 6. Flatten back to PyG format: (Batch * num_nodes, out_channels)
        x = x.view(B * self.num_nodes, -1)

        return x

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
    log_file = '../../Feature_wise_Normalization/log_data_normalization_IT06_knn.log'
    data_file = '../../Feature_wise_Normalization/Pack_all_data.json'
    approach = 'approach1'

    excel_file = '../../clustering_WEMAC_123feat_v03feb.csv'

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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        tf_encoder_bvp = NodeTransformer(in_channels=1, d_model=32, out_channels=7, num_nodes=92).to(device)
        tf_encoder_gsr = NodeTransformer(in_channels=1, d_model=32, out_channels=6, num_nodes=41).to(device)
        tf_encoder_skt = NodeTransformer(in_channels=1, d_model=32, out_channels=2, num_nodes=8).to(device)

        print("All Transformer models have finished training.")

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

        epochs = 5
        combined_model = CombinedGCN_cnn_lstm(tf_encoder_bvp, tf_encoder_gsr, tf_encoder_skt, out_channels=906,
                                              num_classes=2, batch_size=batch_size_all, lstm_hidden=141).to(device)
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

                    # 分别计算三个 Encoder 的 MACs 和 Params
                    # thop.profile 允许直接对子模块进行测试
                    macs_bvp, params_bvp = profile(encoder_bvp, inputs=(x_bvp, edge_index_bvp), verbose=False)
                    macs_gsr, params_gsr = profile(encoder_gsr, inputs=(x_gsr, edge_index_gsr), verbose=False)
                    macs_skt, params_skt = profile(encoder_skt, inputs=(x_skt, edge_index_skt), verbose=False)

                    # 汇总所有的 Encoders 计算量
                    total_encoder_macs = macs_bvp + macs_gsr + macs_skt
                    total_encoder_params = params_bvp + params_gsr + params_skt

                    # FLOPs 通常约等于 MACs 的 2 倍
                    encoder_flops = total_encoder_macs * 2.0
                    encoder_flops_per_sample = encoder_flops / current_batch_size

                    # 格式化输出
                    flops_str, params_str = clever_format([encoder_flops, total_encoder_params], "%.3f")
                    flops_per_sample_str, _ = clever_format([encoder_flops_per_sample, total_encoder_params], "%.3f")

                    print("\n" + "🔥" * 25)
                    print("📊 ENCODER ONLY Complexity Analysis")
                    print("🔥" * 25)
                    print(f"Encoder Parameters : {params_str}")
                    print(f"Encoder FLOPs (BS={current_batch_size}) : {flops_str}")
                    print(f"Encoder FLOPs Per Sample : {flops_per_sample_str}")
                    print("=" * 50 + "\n")
                except Exception as e:
                    print(f"\n[Warning] FLOPs calculation failed: {e}")
                    print("Please ensure 'thop' is installed: pip install thop\n")

        # save
        torch.save(combined_model, "transformer_model.pth")