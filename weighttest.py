import os
from predicts_utils import read_tiff_img
import numpy as np

data_home = '../../data/potsdam/'
labels_path = data_home + 'labels/'
label_files = os.listdir(labels_path)

all = []
for l in label_files:
    label,_ = read_tiff_img(labels_path + l)
    label = np.unique(label)
    for i in label:
        all.append(i)
print(np.unique(np.array(all)))
