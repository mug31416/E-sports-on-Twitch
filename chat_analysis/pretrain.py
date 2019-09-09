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

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INTERVAL = 1000
LR = 5e-3
EPOCHS = 20
BATCH_SIZE = 1


def train(loader, model, optimizer, criterion):
    model.train()
    model.to(DEVICE)
    total_loss = 0.0
    for batch_idx, (inputs, targets, names) in enumerate(loader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE).long()
        preds = model(inputs)
        loss = criterion(preds, targets)
        total_loss += loss.cpu().data.numpy()
        optimizer.zero_grad()
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
    return total_loss / batch_idx, accuracy / total


def main():
    train_loader, val_loader, all_loader = get_loader(log_file, batch_size=BATCH_SIZE)
    # model = MLP(in_dim=500, hidden_dim=[64, 32], out_dim=10)
    model = CNN(drop_out_p=0.1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    loss = 1e5
    model.load_state_dict(torch.load("models/CNN2_2.1778715419760974.pt"))
    for e in range(EPOCHS):
        l = train(train_loader, model, optimizer, criterion)
        if e % 1 == 0:
            _, train_acc = eval(train_loader, model, criterion)
            eval_loss, eval_acc = eval(val_loader, model, criterion)
            print("Epoch {} | Train loss: {} | Train acc: {} | Eval loss: {} | Eval acc: {}"
                  .format(e, l, train_acc, eval_loss, eval_acc))
            # torch.save(model.state_dict(), "models/CNN2_" + str(l) + ".pt")
        # if l < loss:
        #     loss = l
        #     torch.save(model.state_dict(), "models/MLP_" + str(loss) + ".pt")
    # torch.save(model.state_dict(), "models/CNN_" + str(l) + ".pt")
    print("done!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py filename")
        exit(1)
    log_file = sys.argv[1]
    train_split = "train_small.csv"
    valid_split = "valid_small.csv"
    main()
