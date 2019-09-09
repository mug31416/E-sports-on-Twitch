import random, sys

assert len(sys.argv) == 2

f = open("esea_streamers.csv", "r")
lines = f.readlines()[1:]
random.shuffle(lines)

N_PROG = int(sys.argv[1])

wf_lst = []
for i in range(N_PROG):
	wf_lst.append(open("userlist%d.txt" % (i+1), "w"))

for i, line in enumerate(lines):
	user_id, username = line.strip().split(",")
	wf_lst[i % N_PROG].write("%s\n" % username.strip())

for i in range(N_PROG):
	wf_lst[i].close()

f.close()
