import os, sys, librosa, numpy as np, pickle, time
PATH = "../../audio"
file_lst = [fname for fname in os.listdir(PATH) if ".wav" in fname]
output = "../../mfcc"

for fname in file_lst:
	start_time = time.time()
	y, sr = librosa.load(os.path.join(PATH, fname), sr=2000)
	mfcc = librosa.feature.mfcc(y=y, sr=2000)
	with open(os.path.join(output, fname), "wb") as handle:
		pickle.dump(mfcc, handle, protocol=2)

	print (fname, "done. --- %s seconds ---" % (time.time() - start_time))
