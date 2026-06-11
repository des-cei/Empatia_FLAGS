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
@Time : Created in 10:35 AM 2024/11/07
@FileName: CombinedGCN.py
@Software: PyCharm


Introduction of this File:
This file include the combined gcn and cnn model, which take three trained gcn model as input as will as some labeled
data, train a classifier and output the result.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


class CombinedGCN_cnn_lstm(torch.nn.Module):
    def __init__(self, gcn1, gcn2, gcn3, out_channels, num_classes, batch_size, lstm_hidden, lstm_layers=1):
        super(CombinedGCN_cnn_lstm, self).__init__()

        self.gcn1 = gcn1
        self.gcn2 = gcn2
        self.gcn3 = gcn3
        self.batch_size = batch_size

        freeze_model(self.gcn1)
        freeze_model(self.gcn2)
        freeze_model(self.gcn3)

        self.conv1 = nn.Conv1d(in_channels=92, out_channels=92, kernel_size=7, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(92)
        self.conv2 = nn.Conv1d(in_channels=41, out_channels=41, kernel_size=6, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(41)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=2, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(8)

        self.lstm = nn.LSTM(input_size=141, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True)

        self.dense_layer_fc = nn.Sequential(
            nn.Linear(lstm_hidden, num_classes),
            nn.Sigmoid()
        )

        self.fc = torch.nn.Linear(out_channels, num_classes)

    def forward(self, x1, edge_index1, x2, edge_index2, x3, edge_index3):
        out1 = self.gcn1(x1, edge_index1)
        out2 = self.gcn2(x2, edge_index2)
        out3 = self.gcn3(x3, edge_index3)

        out1_reshaped = out1.view(self.batch_size, int(out1.shape[0] / self.batch_size), out1.shape[1])
        out2_reshaped = out2.view(self.batch_size, int(out2.shape[0] / self.batch_size), out2.shape[1])
        out3_reshaped = out3.view(self.batch_size, int(out3.shape[0] / self.batch_size), out3.shape[1])

        out1_cnn = self.bn1(self.conv1(out1_reshaped).squeeze(1))
        out2_cnn = self.bn2(self.conv2(out2_reshaped).squeeze(1))
        out3_cnn = self.bn3(self.conv3(out3_reshaped).squeeze(1))

        output = torch.cat([out1_cnn, out2_cnn, out3_cnn], dim=1).squeeze(2)

        lstm_out, _ = self.lstm(output)

        # lstm_out = lstm_out[:, -1, :]

        output = self.dense_layer_fc(lstm_out)

        return output