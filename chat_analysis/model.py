import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.drop_out = nn.Dropout(p=0.5)
        self.fc = []
        for i, d in enumerate(hidden_dim):
            if i == 0:
                self.fc.append((nn.Sequential(
                    nn.Linear(in_dim, hidden_dim[i]),
                    nn.ReLU()
                )))
            else:
                self.fc.append((nn.Sequential(
                    nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                    nn.ReLU()
                )))

        self.fc.append(nn.Linear(hidden_dim[-1], out_dim))
        self.fc = nn.Sequential(*self.fc)
        self.init_weights()

    def forward(self, input):
        output = self.drop_out(self.fc(input))
        return output

    def init_weights(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight.data)


class CNN(nn.Module):
    def __init__(self, embed_wgt=None, is_time=False, in_dim=500, embed_dim=500, hidden_dim=256, out_dim=10, drop_out_p=0.1):
        super(CNN, self).__init__()
        filter_sizes = [1, 2, 3]
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.is_time = is_time
        # self.in_dim, self.embed_dim = embed_wgt.size()
        # self.embed = nn.Embedding(self.in_dim, self.embed_dim).from_pretrained(embed_wgt, freeze=False)

        if is_time:
            self.embed = nn.Embedding(self.in_dim, self.embed_dim)

        self.conv0 = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.hidden_dim, kernel_size=filter_sizes[0], stride=filter_sizes[0], padding=1),
            nn.LeakyReLU())
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.hidden_dim, kernel_size=filter_sizes[1], stride=filter_sizes[1], padding=1),
            nn.LeakyReLU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.hidden_dim, kernel_size=filter_sizes[2], stride=filter_sizes[2], padding=1),
            nn.LeakyReLU())

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.drop_out = nn.Dropout(drop_out_p)
        self.fc = nn.Linear(2 * 3 * self.hidden_dim, self.out_dim, True)
        self.init_weight()

    def forward(self, input):
        if self.is_time:
            input = self.embed(input).permute(0, 2, 1)
        else:
            input = input.permute(0, 2, 1)
        conv0 = self.conv0(input)
        conv1 = self.conv1(input)
        conv2 = self.conv2(input)

        conv_max0 = self.max_pool(conv0)
        conv_avg0 = self.avg_pool(conv0)
        conv_max1 = self.max_pool(conv1)
        conv_avg1 = self.avg_pool(conv1)
        conv_max2 = self.max_pool(conv2)
        conv_avg2 = self.avg_pool(conv2)

        concat = torch.cat([conv_max0, conv_avg0, conv_max1, conv_avg1, conv_max2, conv_avg2], dim=1)
        concat_drop = self.drop_out(concat)
        out = self.fc(concat_drop.squeeze(0).permute(1, 0))
        return out

    def init_weight(self):
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv1d:
                torch.nn.init.xavier_normal_(m.weight.data)
