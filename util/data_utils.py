import cv2, os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from libtiff import TIFF
# from preprocess import tif16_to_tif8

def read_tiff_img(address):
    """
    读取指定路径的tiff文件，并返回对应numpy array和图像metadata
    :param address:
    :return:
    """
    with rasterio.open(address) as f:
        metadata = f.profile
        result = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
        f.close()
    return result, metadata

class DataSet():
    '''
    通过读入文件生成数据集
    '''

    def __init__(self, image_path, label_path):

        self.image_path = np.array(image_path)
        self.label_path = np.array(label_path)
        self.batch_count = 0
        self.epoch_count = 0

    def num_examples(self):
        '''
        得到样本的数量
        :return:
        '''

        return self.image_path.shape[0]


    def next_batch(self, batch_size):
        '''
        next_batch函数
        :param batch_size:
        :return:
        '''

        start = self.batch_count * batch_size
        end = start + batch_size
        self.batch_count += 1

        if end > self.image_path.shape[0]:
            self.batch_count = 0
            random_index = np.random.permutation(self.image_path.shape[0])
            self.image_path = self.image_path[random_index]
            self.label_path = self.label_path[random_index]
            self.epoch_count += 1
            start = self.batch_count * batch_size
            end = start + batch_size
            self.batch_count += 1

        image_batch, label_batch = self.read_path(self.image_path[start:end],
                                                  self.label_path[start:end])
        return image_batch, label_batch

    def read_path(self, x_path, y_path):
        '''
        将路径读为图片
        :param x_path:
        :param y_path:
        :return:
        '''
        x = []
        y = []
        for i in range(x_path.shape[0]):
            image,_ = read_tiff_img(x_path[i])
            x.append(image)
            # x.append(tif16_to_tif8(TIFF.open(x_path[i], mode='r').read_image()))
            # x.append(self.transform(cv2.imread(x_path[i], cv2.CAP_MODE_RGB)))
            y.append(cv2.imread(y_path[i], cv2.CAP_OPENNI_GRAY_IMAGE))

        # for i in range(len(x)):
        #     print(x[i].shape, len(x[i]))
        #     print(y[i].shape, len(y[i]))

        return np.array(x), np.array(y)

    def transform(self, img):
        """
        :param img:
        :return:
        """

        return img