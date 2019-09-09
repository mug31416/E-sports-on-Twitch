import sys
import csv
from allennlp.modules.elmo import Elmo, batch_to_ids
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# from model import ClsMLP
import fastText
from fastText import train_unsupervised
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from utils import load_followers
from dataloader import get_loader
from model import MLP, CNN

import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INTERVAL = 100
LR = 1e-6
EPOCHS = 10
BATCH_SIZE = 1

parser = argparse.ArgumentParser(description='Train on chat text data.')
parser.add_argument('--pretrain', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help="whether it is pretrain or not.")
parser.add_argument('--is_time', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="whether it is using temporal feature or not.")
parser.add_argument('--prediction', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="whether it is for prediction or not.")
parser.add_argument('--logfile', default="chat_log_target.csv", type=str,
                    help="filename for the log file")
parser.add_argument('--triplet_loss', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="whether to use triplet loss or not.")
args = parser.parse_args()


class TripletMarginLoss(nn.Module):
    """Triplet loss function.
    Based on: http://docs.chainer.org/en/stable/_modules/chainer/functions/loss/triplet.html
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist = torch.sum((anchor - positive) ** 2 - (anchor - negative) ** 2, dim=1) + self.margin
        dist_hinge = torch.clamp(dist, min=0.0)  # maximum between 'dist' and 0.0
        loss = torch.mean(dist_hinge)
        return loss


def triplet_loss(input1, input2, input3, margin=1.0):
    return TripletMarginLoss(margin)(input1, input2, input3)


def train(loader, model, optimizer, criterion):
    model.train()
    model.to(DEVICE)
    total_loss = 0.0
    print("training", len(loader))
    for batch_idx, (inputs, targets, names) in enumerate(loader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE).long()
        preds = model(inputs)
        # print(F.softmax(preds))
        loss = criterion(preds, targets)
        total_loss += loss.cpu().data.numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % INTERVAL == INTERVAL - 1:
            print("Batch [{}/{}] | Loss: {}".format(batch_idx, len(loader), total_loss / batch_idx))
    return total_loss / batch_idx


def train_triplet_loss(loader, model, optimizer):
    model.train()
    model.to(DEVICE)
    total_loss = 0.0
    for batch_idx, (data_a, data_p, data_n) in enumerate(loader):
        data_a, data_p, data_n = data_a.to(DEVICE), data_p.to(DEVICE), data_n.to(DEVICE)
        optimizer.zero_grad()
        pred_a, pred_p, pred_n = model(data_a), model(data_p), model(data_n)
        loss = triplet_loss(pred_a, pred_p, pred_n)
        total_loss += loss.cpu().data.numpy()
        loss.backward()
        optimizer.step()
        if batch_idx % INTERVAL == INTERVAL - 1:
            print("Batch [{}/{}] | Loss: {}".format(batch_idx, len(loader), total_loss / batch_idx))
    return total_loss / batch_idx


def eval(loader, model, criterion):
    model.eval()
    model.to(DEVICE)
    accuracy = 0
    total = 0
    total_loss = 0.0

    for batch_idx, (inputs, targets, names) in enumerate(loader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE).long()
        preds = model(inputs)
        loss = criterion(preds, targets)
        total_loss += loss.cpu().data.numpy()
        _, pred_labels = torch.max(F.softmax(preds, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        accuracy += torch.sum(torch.eq(pred_labels, targets)).item()
        total += len(targets)
    print("batch_idx", batch_idx)
    return total_loss / batch_idx, accuracy / total


def generate_score(loader, model, output_file):
    model.eval()
    model.to(DEVICE)

    fwrite = open(output_file, "w")
    fwrite.write("user,prob\n")
    # for i, prob in enumerate(y_prob):
    #     fwrite.write(username_val[i] + "," + str(prob) + "\n")
    accuracy = 0
    total = 0
    for batch_idx, (inputs, targets, names) in enumerate(loader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE).long()
        preds = model(inputs)
        _, pred_labels = torch.max(F.softmax(preds, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        accuracy += torch.sum(torch.eq(pred_labels, targets)).item()
        total += len(targets)
        preds = F.softmax(preds, dim=1)[:, 1].detach().numpy()
        for i, prob in enumerate(preds):
            fwrite.write(names[i] + "," + str(prob) + "\n")
    fwrite.close()
    print("generate accuracy", accuracy / total)
    return


def main():
    if args.pretrain:
        print("Pretraining...")
    else:
        print("On target dataset...")
    print("The model is for time: {}.".format(args.is_time))
    if args.pretrain:
        train_loader, val_loader, all_loader = get_loader(log_file,
                                                          batch_size=BATCH_SIZE,
                                                          pretrain=args.pretrain,
                                                          is_time=args.is_time,
                                                          triplet_loss=args.triplet_loss)
    else:
        train_loader, val_loader = get_loader(log_file,
                                              batch_size=BATCH_SIZE,
                                              pretrain=args.pretrain,
                                              is_time=args.is_time,
                                              triplet_loss=args.triplet_loss)
    # model = MLP(in_dim=500, hidden_dim=[64, 32], out_dim=10)
    if not args.is_time:
        model = CNN(drop_out_p=0.2 if args.pretrain else 0.4, out_dim=10 if args.pretrain else 2)
    else:
        model = CNN(drop_out_p=0.2 if args.pretrain else 0.5, out_dim=10 if args.pretrain else 2, is_time=args.is_time,
                    in_dim=50000, embed_dim=100)
    if not args.prediction:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)

        loss = 1e5
        # model.load_state_dict(torch.load("models/target_CNN2_temporal2.027013686898257.pt"))
        # model.out = nn.Linear(2 * 3 * 256, 2, True)
        # for param in model.parameters():
        #     param.requires_grad = True
        for e in range(EPOCHS):
            if e > 1:
                for param in model.parameters():
                    param.requires_grad = True
            if not args.triplet_loss:
                l = train(train_loader, model, optimizer, criterion)
            else:
                l = train_triplet_loss(train_loader, model, optimizer)
            if e % 1 == 0:
                if not args.triplet_loss:
                    _, train_acc = eval(train_loader, model, criterion)
                eval_loss, eval_acc = eval(val_loader, model, criterion)
                print("Epoch {} | Train loss: {} | Train acc: {} | Eval loss: {} | Eval acc: {}"
                      .format(e, l, train_acc if not args.triplet_loss else "N/A", eval_loss, eval_acc))
                if not args.is_time:
                    torch.save(model.state_dict(), "models/nopretrain/triplet_loss/text" + str(l) + ".pt")
                else:
                    torch.save(model.state_dict(), "models/nopretrain/triplet_loss/temporal" + str(l) + ".pt")
    else:
        model.load_state_dict(torch.load("models/nopretrain/triplet_loss/temporal0.857176048699582.pt"))
        generate_score(train_loader, model, "chat_train_nopretrain_temporal.csv")
        generate_score(val_loader, model, "chat_val_nopretrain_temporal.csv")
    print("done!")


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python train.py filename")
    #     exit(1)
    # log_file = sys.argv[1]
    log_file = args.logfile
    train_split = "train_small.csv"
    valid_split = "valid_small.csv"
    main()
