import os, sys

f = open("train_small.csv", "r")
train = [l.strip().split(",") for l in f.readlines()[1:]]
f.close()

f = open("valid_small.csv", "r")
val = [l.strip().split(",") for l in f.readlines()[1:]]
f.close()

t = open("train.csv", "w")
t.write("TwitchID,isRankA\n")
for player, rank_A, _, _, video, _ in train:
	player = player[1:-1]
	if int(video) > 0:
		t.write("%s,%s\n" % (player, rank_A))
t.close()

t = open("val.csv", "w")
t.write("TwitchID,isRankA\n")
for player, rank_A, _, _, video, _ in val:
	player = player[1:-1]
	if int(video) > 0:
		t.write("%s,%s\n" % (player, rank_A))
t.close()
