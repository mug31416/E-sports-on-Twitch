import os, sys

f = open("supp.txt", "r")
users = [line.strip() for line in f.readlines()]
f.close()

ALL_VIDEO = "../../video/"
S3 = "s3://11775projecttwitchvideostore/twitch-5min/"

for u in users:
	print (u)
	src = S3
	dest = ALL_VIDEO
	os.system("aws s3 cp '%s' '%s' --recursive --exclude '*' --include '%s*'" % (src, dest, u))
