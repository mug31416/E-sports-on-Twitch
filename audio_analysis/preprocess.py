import os, sys

ALL_VIDEO = "../../video/"
VIDEO = "../../sample_video/"
AUDIO = "../../audio/"

f = open("train_and_val.csv", "r")
users = [line.strip().split(",")[0] for line in f.readlines()[1:]]
f.close()

all_vids = os.listdir(ALL_VIDEO)
for user in users:
	matches = [vid for vid in all_vids if vid.startswith(user)]
	if not matches:
		print ("not found any video for user %s" % user)
		continue

	src = os.path.join(ALL_VIDEO, matches[0])
	dest = os.path.join(VIDEO, matches[0])
	os.system("cp '%s' '%s'" % (src, dest))
		
