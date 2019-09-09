import os, subprocess

src = "../../audio/"
dest = "../../SoundNet-tensorflow/data"

for fname in os.listdir(src):

	src_file = os.path.join(src, fname)
	dest_file = os.path.join(dest, fname[:-3] + "wav")
	subprocess.call([
		"cp",
		src_file,
		dest_file,
	])

f = open("wav_audio.txt", "w")
g = open("audio.txt", "r")
for x in g.readlines():
	f.write("%s\n" % g.strip()[:-3] + "wav")

f.close()
g.close()
