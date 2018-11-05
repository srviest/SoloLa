
import numpy as np

asc_path = "models/cnn_normmc/ascending.npz"
des_path = "models/cnn_normmc/descending.npz"
model_fp = asc_path
npzfile = np.load(model_fp,encoding="latin1")

print(npzfile.files)
for k in npzfile.iterkeys():
    print("------")
    print(k)
    print(npzfile[k])
