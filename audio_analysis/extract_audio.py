import os, subprocess

"""
ffmpeg -y -i videofile -ac 1 -f wav audiofile

"""

PATH = "../../sample_video/"
AUDIO = "../../SoundNet-tensorflow/data/"

print ("Total number of videos:", len(os.listdir(PATH)))
for video in os.listdir(PATH):
	videofile = os.path.join(PATH, video)
	audiofile = os.path.join(AUDIO, video[:-1]+"3")
	subprocess.call([
		'ffmpeg',
		'-i',
		videofile,
		# '-q:a'
		# '0',
		# '-map',
		# 'a',
		audiofile,
	])

	print ("%s extracted" % audiofile)


