from torch.utils.data import Dataset, DataLoader
from utils import parse_log, load_followers, convert_followers_to_class
import numpy as np
import csv
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from extract_feat import generate_one_hour_density_feature


class User:
    def __init__(self, username, message, followers=None, rank=None):
        self.username = username
        self.message = message
        self.rank = rank
        self.followers = followers


class ChatData(Dataset):
    def __init__(self, X, y, names):
        self.X = X
        self.y = y
        self.names = names

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item], self.names[item]


class TripletChatData(ChatData):
    """Only used for train loader"""
    def __init__(self, X, y, names):
        super(TripletChatData, self).__init__(X, y, names)
        self.n_triplets = 10000
        self.triplets = self.generate_triplets(self.y)

    def generate_triplets(self, labels):
        triplets = []
        for x in range(self.n_triplets):
            idx = np.random.randint(0, len(labels))
            idx_matches = np.where(np.array(labels) == labels[idx])[0]
            idx_no_matches = np.where(np.array(labels) != labels[idx])[0]
            idx_a, idx_p = np.random.choice(idx_matches, 2, replace=False)
            idx_n = np.random.choice(idx_no_matches, 1)[0]
            triplets.append([idx_a, idx_p, idx_n])
        return np.array(triplets)

    def __len__(self):
        return self.triplets.shape[0]

    def __getitem__(self, item):
        t = self.triplets[item]
        a, p, n = self.X[t[0]], self.X[t[1]], self.X[t[2]]
        return a, p, n


def load_feat_data(data, pretrain):
    print("loading feature data...")
    if pretrain:
        feat_path = "fasttext_feat_500_sent/"
    else:
        feat_path = "fasttext_feat_500_sent_target/"
    X = []
    y = []
    names = []
    for username in data:
        try:
            feat = np.load(feat_path + username + ".npy")
        except:
            continue
        if len(feat) > 50:
            feat = np.random.choice(feat, 50, replace=False)
        for c in feat:
            X.append(c)
            if pretrain:
                y.append(data[username].followers)
            else:
                y.append(data[username].rank == 'A')
            names.append(username)
    if pretrain:
        y = convert_followers_to_class(y, num_class=10)
    assert len(X) == len(y)
    return X, y, names


def load_data(user_log, user_followers, pretrain, split=None, output_file=None):
    print("loading user chat logs and followers...")
    data = {}
    if output_file is not None:
        data_file = open(output_file, "w")
    if pretrain:
        for username in user_log:
            messages = user_log[username]
            if username in user_followers:
                data[username] = User(username, messages, followers=user_followers[username])
                if output_file is not None:
                    for m in messages:
                        data_file.write(m.content + "\n")
    else:
        assert split is not None
        with open(split, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)  # skip header: twitch,rank_A,rank,num_days_chat,num_days_video,followers_03
            for log in csv_reader:
                username = log[0]
                if username not in user_log:
                    continue
                rank_A = int(log[1])  # 0 or 1
                rank = log[2]  # A, B, C, D
                messages = user_log[username]
                data[username] = User(username, messages, rank=rank)
                if output_file is not None:
                    for m in messages:
                        data_file.write(m + "\n")
    print("loaded!")
    return data


def load_temporal_feat(user_log, user_followers, pretrain, split=None):
    user_density = generate_one_hour_density_feature(user_log)
    X = []
    y = []
    names = []
    if pretrain:
        for username in user_density:
            feat = np.array(user_density[username])
            if username in user_followers:
                X.append(feat)
                y.append(user_followers[username])
                y = convert_followers_to_class(y, num_class=10)
                names.append(username)
    else:
        assert split is not None
        with open(split, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)
            for log in csv_reader:
                username = log[0]
                if username not in user_log:
                    continue
                rank_A = int(log[1])
                X.append(np.array(user_density[username]))
                y.append(rank_A)
                names.append(username)
    return X, y, names


def get_dataset(log_file, pretrain=True, is_time=False, triplet_loss=False):
    user_log = parse_log(log_file=log_file)  # {username: [Message(content, timestamp)]}
    user_followers = load_followers("followers.csv")
    if pretrain:
        if not is_time:
            train_data = load_data(user_log, user_followers, pretrain)
            X_all, y_all, names_all = load_feat_data(train_data, pretrain=pretrain)
        else:
            X_all, y_all, names_all = load_temporal_feat(user_log, user_followers, pretrain)

        X_train, X_val, y_train, y_val, names_train, names_val = train_test_split(X_all, y_all, names_all, test_size=0.25, random_state=42)
        print("train name size", len(set(names_train)))
        print("val name size", len(set(names_val)))
        if triplet_loss:
            trainset = TripletChatData(X_train, y_train, names_train)
        else:
            trainset = ChatData(X_train, y_train, names_train)
        valset = ChatData(X_val, y_val, names_val)
        allset = ChatData(X_all, y_all, names_all)
        return trainset, valset, allset
    else:
        train_split = "train_small.csv"
        valid_split = "valid_small.csv"
        if not is_time:
            train_data = load_data(user_log, user_followers, pretrain, split=train_split)
            val_data = load_data(user_log, user_followers, pretrain, split=valid_split)
            X_train, y_train, names_train = load_feat_data(train_data, pretrain=pretrain)
            X_val, y_val, names_val = load_feat_data(val_data, pretrain=pretrain)
        else:
            X_train, y_train, names_train = load_temporal_feat(user_log, user_followers, pretrain, split=train_split)
            X_val, y_val, names_val = load_temporal_feat(user_log, user_followers, pretrain, split=valid_split)
        if triplet_loss:
            trainset = TripletChatData(X_train, y_train, names_train)
        else:
            trainset = ChatData(X_train, y_train, names_train)
        valset = ChatData(X_val, y_val, names_val)
        return trainset, valset


def get_loader(log_file, batch_size=1, pretrain=True, is_time=False, triplet_loss=False):
    if pretrain:
        trainset, valset, allset = get_dataset(log_file=log_file, pretrain=pretrain, is_time=is_time,
                                               triplet_loss=triplet_loss)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
        all_loader = DataLoader(allset, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader, all_loader
    else:
        trainset, valset = get_dataset(log_file=log_file, pretrain=pretrain, is_time=is_time,
                                       triplet_loss=triplet_loss)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader


if __name__ == "__main__":
    get_dataset("chat_log_pretrain.csv")
