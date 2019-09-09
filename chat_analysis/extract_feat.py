import random
import torch
import numpy as np
import torch.nn as nn
import fastText
from fastText import train_unsupervised
from allennlp.modules.elmo import Elmo, batch_to_ids
from nltk.tokenize import word_tokenize

from utils import parse_log

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


def load_data(user_log, output_file=None):
    print("loading user chat logs and followers...")
    data = {}
    if output_file is not None:
        data_file = open(output_file, "w")
    for username in user_log:
        messages = user_log[username]
        if output_file is not None:
            for m in messages:
                data_file.write(m.content + "<eos>" + "\n")
    print("loaded!")
    return data


def generate_fasttext_embeddings(data, model):
    print("generating fasttext embeddings...")
    max_interval = 60 * 60  # one hour
    for user in data:
        print(user)
        messages = data[user]
        messages.sort()
        # sent = " ".join(messages)
        # feat_vector = model.get_sentence_vector(sent)
        feat_vector = []
        if len(messages) < 1:
            continue
        prev_time = messages[0].timestamp
        curr_sent = []
        for m in messages:
            interval = m.timestamp - prev_time
            if interval <= max_interval:
                curr_sent.append(m.content)
            else:
                sent_embed = []
                for sent in curr_sent:
                    sent_embed.append(model.get_sentence_vector(sent))
                feat_vector.append(np.array(sent_embed))
                curr_sent = []
                curr_sent.append(m.content)
            prev_time = m.timestamp
        sent_embed = []
        for sent in curr_sent:
            sent_embed.append(model.get_sentence_vector(sent))
        feat_vector.append(np.array(sent_embed))
        np.save(feat_path + user, feat_vector)
        # print(user, len(messages), len(feat_vector))
    print("fasttext feature generated!")
    return


def generate_one_hour_density_feature(data):
    user_density = {}
    max_interval = 24 * 60 * 60  # one day
    for user in data:
        messages = data[user]
        messages.sort()
        if len(messages) < 1:
            continue
        prev_time = messages[0].timestamp
        curr_cnt = 0
        feat_vector = []
        for m in messages:
            interval = m.timestamp - prev_time
            if interval <= max_interval:
                curr_cnt += 1
            else:
                feat_vector.append(curr_cnt)
                curr_cnt = 1
            prev_time = m.timestamp
        feat_vector.append(curr_cnt)
        # print(user, len(feat_vector))
        user_density[user] = feat_vector
    return user_density


def calculate_chat_density(user_time):
    user_density = {}
    for username in user_time:
        times = user_time[username]
        if max(times) != min(times):
            density = len(times) / (max(times) - min(times))
            if density > 1e-2:
                continue
            user_density[username] = density * 1e5
            print(username, user_density[username])
    return user_density


if __name__ == "__main__":
    user_log = parse_log("chat_log_target.csv")
    # load_data(user_log, output_file="pretrain_data.txt")
    # print("generating model...")
    # model = train_unsupervised(input="pretrain_data.txt", model='skipgram', dim=500)
    # model.save_model("pretrain_token.bin")

    model = fastText.load_model("pretrain_token.bin")
    feat_path = "fasttext_feat_500_sent_target/"
    generate_fasttext_embeddings(user_log, model)
