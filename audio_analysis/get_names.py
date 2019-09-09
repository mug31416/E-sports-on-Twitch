import os, sys

folder = sys.argv[1]
dest = open(sys.argv[2], "w")

for fname in os.listdir(folder):
	dest.write("./data/%s\n" % fname.strip())

dest.close()
