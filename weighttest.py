<<<<<<< HEAD
=======
import os
from util.predicts_utils import read_tiff_img
import numpy as np
>>>>>>> 4c0e5a013738e536521711a5d9794385e17a599a

import numpy as np
import tensorflow as tf

label = np.random.randint(0, 5, size=[5, 5, 5])
print(label)
print(label.shape)
label = tf.one_hot(label, depth=6).numpy()
print(label)
print(label.shape)
label = np.reshape(label, (label.shape[0]*label.shape[1]*label.shape[2], label.shape[3]))
label = [np.argmax(i) for i in label]
label = np.array(label)
print(label)
print(label.shape)
label = np.reshape(label, [5, 5, 5])
print(label)
print(label.shape)
