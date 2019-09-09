import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Rank:
    def __init__(self, position, rank_group):
        self.position = position
        self.rank_group = rank_group

    def __str__(self):
        return "position: " + str(self.position) + ", rank_group: " + self.rank_group


class Chat:
    def __init__(self, sender, message):
        self.sender = sender
        self.message = message

    def __str__(self):
        line = "[" + self.sender + "] : "
        line += self.message
        return line


if __name__ == "__main__":

    with open('player_list.json') as file:
        player_list = json.load(file)
        file.close()

    leaderboards = {}  # {month:{id:Rank}}
    leaderboards['01'] = {}
    leaderboards['02'] = {}

    for player in player_list:
        pid = player.split('/')[-1]
        ranks = player_list[player]
        for rank in ranks:
            month = rank.split('-')[-1]  # '01' or '02'
            position = int(ranks[rank]['position'])
            rank_group = ranks[rank]['rank_group']
            rank = Rank(position, rank_group)
            leaderboards[month][pid] = rank

    player_info = {}  # {name:id}
    with open('esea_streamers.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        header = next(csv_reader, None)
        for line in csv_reader:
            id = line[0]
            name = line[1]
            player_info[name] = id
        file.close()

    chat_log = {}  # {id, [Chat]}
    with open('chat_log.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        header = next(csv_reader, None)
        for line in csv_reader:
            channel = line[1]
            name = channel[1:]
            id = player_info[name]
            sender = line[2]
            message = line[3]
            chat = Chat(sender, message)
            if id not in chat_log:
                chat_log[id] = []
            chat_log[id].append(chat)

    # player_chat = {}  # {id:#chat}
    # chat_number = []
    # position_01 = []
    # position_02 = []
    # for id in chat_log:
    #     player_chat[id] = len(chat_log[id])
    #     chat_number.append(len(chat_log[id]))
    #     if id in leaderboards['01']:
    #         rank_01 = leaderboards['01'][id]

    #         position_01.append(rank_01.position)
    # plt.scatter(chat_number, position_01)
    # plt.show()
    # sns.set_style('darkgrid')
    # ax = sns.distplot(chat_number, kde=False, rug=False, bins=10)
    # ax.set(xlabel='Number of chats per channel', ylabel='Number of channels')
    # plt.show()




