import csv
import copy
from bisect import bisect_right
import matplotlib.pyplot as plt
# from twitch import TwitchClient

# client = TwitchClient('7wf5a07zm99kh3komhy2yoodiub1ns')


class Message:
    def __init__(self, content, timestamp):
        self.content = content
        self.timestamp = timestamp

    def __lt__(self, other):
        return self.timestamp < other.timestamp


def parse_log(log_file):
    print("parsing chat log...")
    user_log = {}
    with open(log_file, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # skip header: id,channel,sender,message,date
        for log in csv_reader:
            channel = log[1]
            username = channel[1:]
            message = log[3]
            timestamp = int(log[4])
            if username not in user_log:
                user_log[username] = []
            user_log[username].append(Message(message, timestamp))
        csv_file.close()
    print("chat log parsed!")
    return user_log


def get_unique_players_from_log(log_file, out_file=None):
    user_log = parse_log(log_file)
    users = []
    for user in user_log:
        users.append(user)
    print("extract {} players from log file [{}]!".format(len(users), log_file))
    if out_file is not None:
        fwrite = open(out_file, "w")
        for user in users:
            fwrite.write(user + "\n")
        print("Write the players to the output file [{}]".format(out_file))
    return users


def get_ids_from_names(usernames):
    i = 0
    userids = {}
    while i < len(usernames):
        users = client.users.translate_usernames_to_ids(usernames[i : i + 100])
        i += 100
        for user in users:
            userids[user.name] = user.id
    return userids


def get_followers_from_ids(userids, out_file="followers.csv"):
    user_followers = {}
    cnt = 0
    fwrite = open(out_file, "w")
    fwrite.write("user,followers\n")

    for user in userids:
        if cnt % 10 == 0:
            print(cnt)
        cnt += 1
        uid = userids[user]
        followers = int(client.channels.get_by_id(uid).followers)
        user_followers[user] = followers
        fwrite.write(user + "," + str(followers) + "\n")
    return user_followers


def load_followers(follower_file):
    user_followers = {}
    with open(follower_file, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # skip header: id,channel,sender,message,date
        for log in csv_reader:
            username = log[0]
            followers = int(log[1])
            user_followers[username] = followers
        csv_file.close()
    print("follower file parsed!")
    return user_followers


def convert_followers_to_class(followers, num_class=10, mode='uniform'):
    thresholds = get_thresholds_from_followers(followers, num_class)
    followers_class = []
    for f in followers:
        c = bisect_right(thresholds, f)
        if c >= len(thresholds):
            c -= 1
        elif thresholds[c - 1] == f:
            c -= 1
        followers_class.append(c)
    # for i in range(num_class):
    #     print(i, followers_class.count(i))
    return followers_class


def get_thresholds_from_followers(followers, num_class=10, mode='uniform'):
    followers_copy = copy.copy(followers)
    followers_copy.sort()
    thresholds = []
    if mode == 'uniform':
        interval = len(followers_copy) // num_class
        for i in range(num_class):
            thresholds.append(followers_copy[min(int((i + 1) * interval), len(followers_copy) - 1)])
    return thresholds


if __name__ == "__main__":
    # users = get_unique_players_from_log("chat_log_pretrain.csv")
    # userids = get_ids_from_names(users)
    # user_followers = get_followers_from_ids(userids)
    user_followers = load_followers("followers.csv")
    followers = []
    for username in user_followers:
        followers.append(user_followers[username])
    convert_followers_to_class(followers)
