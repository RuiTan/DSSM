import cv2
import numpy as np
import random

from libtiff import TIFF
from tqdm import tqdm
import pandas as pd
import os
from util.predicts_utils import read_tiff_img


size = 256
# size = 512
# data_home = 'dataset/scale/potsdam/'
# data_home = 'dataset/noscale/potsdam/'
data_home = '../../data/potsdam/'
# data_home = '../../data/zurich/'

def write_tiff(tiff_file, image):
    out = TIFF.open(tiff_file, mode='w')
    out.write_image(image, compression=None, write_rgb=True)
    out.close()

def generate_train_dataset(image_num=20000,
                           train_image_path=data_home + 'train/images/',
                           train_label_path=data_home + 'train/labels/',
                           test_image_path=data_home + 'test/images/',
                           test_label_path=data_home + 'test/labels/'):

    if not os.path.exists(train_image_path): os.makedirs(train_image_path)
    if not os.path.exists(train_label_path): os.makedirs(train_label_path)
    if not os.path.exists(test_image_path): os.makedirs(test_image_path)
    if not os.path.exists(test_label_path): os.makedirs(test_label_path)

    # 用来记录所有的子图的数目
    g_count = 1
    images_path = os.listdir(data_home + 'images/')
    labels_path = os.listdir(data_home + 'labels/')
    images_path = np.sort(images_path)
    labels_path = np.sort(labels_path)
    print(images_path)
    print(len(images_path))
    print(labels_path)
    print(len(labels_path))
    # for i in images_path:
    #     if i.endswith('.tfw'):
    #         images_path.remove(i)
    # labels_path = []
    # for i in images_path:
    #     if i.endswith('.tif'):
    #         labels_path.append(i.replace('zh', '').replace('tif','png'))

    # 每张图片生成子图的个数
    image_each = image_num // len(images_path)
    image_path, label_path = [], []
    g_random = random.sample(range(1, image_num + 10), image_num)
    for i in tqdm(range(len(images_path))):
        count = 0
        # image = cv2.imread(images_path[i])
        image, _ = read_tiff_img(data_home + 'images/' + images_path[i])
        # image = TIFF.open(images_path[i], mode='r').read_image()
        if (labels_path[0].endswith('tif')):
            label,_ = read_tiff_img(data_home + 'labels/' + labels_path[i])
        else:
            label = cv2.imread(data_home + 'labels/' + labels_path[i], cv2.CAP_OPENNI_GRAY_IMAGE)
        print(np.unique(label))
        print('\nread image: ' + images_path[i])
        print('read label: ' + labels_path[i])
        X_height, X_width = image.shape[0], image.shape[1]
        while count < image_each:
            random_width = random.randint(0, X_width - size - 1)
            random_height = random.randint(0, X_height - size - 1)
            image_ogi = image[random_height: random_height + size, random_width: random_width + size, :]
            label_ogi = label[random_height: random_height + size, random_width: random_width + size]

            image_d, label_d = data_augment(image_ogi, label_ogi)

            if g_random[i * image_each + count] > 100:
                image_path.append(train_image_path + '%05d.tif' % g_count)
                label_path.append(train_label_path + '%05d.png' % g_count)
                write_tiff((train_image_path + '%05d.tif' % g_count), image_d)
                cv2.imwrite((train_label_path + '%05d.png' % g_count), label_d)
            else:
                write_tiff((test_image_path + '%05d.tif' % g_count), image_d)
                cv2.imwrite((test_label_path + '%05d.png' % g_count), label_d)

            count += 1
            g_count += 1
    df = pd.DataFrame({'image':image_path, 'label':label_path})
    df.to_csv(data_home + 'path_list.csv', index=False)

# 以下函数都是一些数据增强的函数
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(size)]

    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)

    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)

    gamma = np.exp(alpha)

    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((size /2, size / 2), angle, 1)

    xb = cv2.warpAffine(xb, M_rotate, (size, size))

    yb = cv2.warpAffine(yb, M_rotate, (size, size))

    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))

    return img


def add_noise(img):
    for i in range(size):  # 添加点噪声

        temp_x = np.random.randint(0, img.shape[0])

        temp_y = np.random.randint(0, img.shape[1])

        img[temp_x][temp_y] = 255

    return img


def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)

    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)

    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)

    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转

        yb = cv2.flip(yb, 1)

    # if np.random.random() < 0.25:
    #     xb = random_gamma_transform(xb, 1.0)

    # if np.random.random() < 0.25:
    #     xb = blur(xb)

    # # 双边过滤
    # if np.random.random() < 0.25:
    #     xb =cv2.bilateralFilter(xb,9,75,75)
    #
    # #  高斯滤波
    # if np.random.random() < 0.25:
    #     xb = cv2.GaussianBlur(xb,(5,5),1.5)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb, yb

if __name__ == '__main__':
    generate_train_dataset(image_num=20000)
