import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import os

from tqdm import tqdm

ds = 'potsdam'

def iou(y_pre: np.ndarray, y_true: np.ndarray) -> 'dict':
    metric_dict = {}
    if ds == 'zurich':
        # cm是混淆矩阵
        cm = confusion_matrix(
            y_true=y_true,
            y_pred=y_pre,
            # labels=[0, 1, 2, 3, 4]
            labels=[0, 1, 2, 3, 4, 5, 6, 7, 8]
        )
        # print(cm)

        result_iou = []

        for i in range(len(cm)):
            if (sum(cm[i, :]) + sum(cm[:, i]) - cm[i, i]) == 0:
                result_iou.append(0)
            else:
                result_iou.append(cm[i][i] / (sum(cm[i, :]) + sum(cm[:, i]) - cm[i, i]))

        metric_dict = {}
        metric_dict['IOU/Null'] = result_iou[0]
        metric_dict['IOU/Road'] = result_iou[1]
        metric_dict['IOU/Bare soil'] = result_iou[2]
        metric_dict['IOU/Tree'] = result_iou[3]
        metric_dict['IOU/Water'] = result_iou[4]
        metric_dict['IOU/Buildings'] = result_iou[5]
        metric_dict['IOU/Grass'] = result_iou[6]
        metric_dict['IOU/Rails'] = result_iou[7]
        metric_dict['IOU/Pool'] = result_iou[8]

        metric_dict['iou'] = np.mean(result_iou)
        metric_dict['accuracy'] = sum(np.diag(cm)) / sum(np.reshape(cm, -1))

    elif ds == 'potsdam':

        # cm是混淆矩阵
        cm = confusion_matrix(
            y_true=y_true,
            y_pred=y_pre,
            labels=[0, 1, 2, 3, 4, 5]
        )
        # print(cm)
        result_iou = []

        for i in range(len(cm)):
            if (sum(cm[i, :]) + sum(cm[:, i]) - cm[i, i]) == 0:
                result_iou.append(0)
            else:
                result_iou.append(cm[i][i] / (sum(cm[i, :]) + sum(cm[:, i]) - cm[i, i]))

        metric_dict = {}
        metric_dict['IOU/Bare Soil'] = result_iou[0]
        metric_dict['IOU/Building'] = result_iou[1]
        metric_dict['IOU/Tree'] = result_iou[2]
        metric_dict['IOU/Car'] = result_iou[3]
        metric_dict['IOU/Grass'] = result_iou[4]
        metric_dict['IOU/Road'] = result_iou[5]

        metric_dict['iou'] = np.mean(result_iou)
        metric_dict['accuracy'] = sum(np.diag(cm)) / sum(np.reshape(cm, -1))
    return metric_dict


def result(labels_path, predicts_path):
    labels = np.sort(os.listdir(labels_path))
    predicts = np.sort(os.listdir(predicts_path))
    if ds == 'potsdam':
        labels = labels[:5]
        predicts = predicts[:5]
    # print(labels)
    # print(predicts)
    y_true = None
    y_predict = None
    # print('read label:')
    for i in labels:
        label = cv2.imread(labels_path + i, cv2.CAP_OPENNI_GRAY_IMAGE)
        print(label.shape)
        if y_true is None:
            y_true = np.reshape(label, -1)
        else:
            y_true = np.append(y_true, np.reshape(label, -1))
    # print('read predict:')
    for i in predicts:
        predict = cv2.imread(predicts_path + i)
        print(predict.shape)
        predict = color_to_label(predict)
        if y_predict is None:
            y_predict = np.reshape(predict, -1)
        else:
            y_predict = np.append(y_predict, np.reshape(predict, -1))
    result = iou(np.reshape(y_predict, -1), np.reshape(y_true, -1))
    for key in result.keys():
        offset = 40 - key.__len__()
        print(key + ' ' * offset + '%.4f' % result[key])

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(int)
    return gray

def color_to_label(img):
    labels = []
    if ds == 'potsdam':
        # labels = [29, 76, 149, 178, 225, 254]
        # labels = [0, 44, 64, 73, 99, 254]
        labels = [254, 0, 44, 73, 64, 99]
    elif ds == 'zurich':
        labels = [254, 0, 44, 73, 64, 99, 149, 181, 178]
    gray = rgb2gray(img)
    # print(np.unique(gray))
    j = 0
    # if ds == 'zurich':
    gray[gray == 0] = 9
    for i in labels:
        if i == 0:
            j += 1
            continue
        gray[gray == i] = j
        j += 1
    # if ds == 'zurich':
    gray[gray == 9] = 1
    # print(np.unique(gray))
    return gray

def label_to_color(img: np.ndarray) -> np.ndarray:
    color = np.ones([img.shape[0], img.shape[1], 3])
    print(img.shape)
    if ds == 'zurich':
        color[img == 0] = [255, 255, 255]  # 白-Null
        color[img == 1] = [0, 0, 0]  # 黑-Road
        color[img == 2] = [150, 0, 0]  # 棕-Bare soil
        color[img == 3] = [0, 125, 0]  # 绿-Tree
        color[img == 4] = [0, 80, 150]  # 蓝-Water
        color[img == 5] = [100, 100, 100]  # 灰-Buildings
        color[img == 6] = [0, 255, 0]  # 青-Grass
        color[img == 7] = [255, 150, 150]  # 黄-Rails
        color[img == 8] = [0, 255, 255]  # 天蓝-Pool
    elif ds == 'potsdam':
        img = np.squeeze(img)
        color[img == 0] = [255, 0, 0]  # 红：裸土
        color[img == 1] = [0, 0, 255]  # 蓝：建筑
        color[img == 2] = [0, 255, 0]  # 绿：树
        color[img == 3] = [255, 255, 0]  # 黄：车
        color[img == 4] = [0, 255, 255]  # 天蓝：草地
        color[img == 5] = [255, 255, 255]  # 白：道路
    return color

def recolor(predicts_path, out_path):
    predicts = np.sort(os.listdir(predicts_path))
    predicts = predicts[:5]
    if not os.path.exists(out_path): os.makedirs(out_path)
    for i in tqdm(predicts):
        predict = cv2.imread(predicts_path + i)
        predict = color_to_label(predict)
        colored = label_to_color(predict)
        cv2.imwrite(out_path + i, colored)


if __name__ == '__main__':
    # models = ['1216_rgbnir-50000', '1216_rgbnir_plus-50000', '1221nir-50000', '1221rgb-50000']
    models = ['1222_rgb-50000', '1222_nir-50000', '1222_rgbnir-50000', '1222_rgbnir_plus-50000', ]
    labels_path = '../../data/' + ds + '/labels/'
    for m in models:
        # print(m)
        predicts_path = '../../data/' + ds + '/result/' + m + '/'
        out_path = '../../data/' + ds + '/result_color/' + m + '/'
        # result(labels_path, predicts_path)
        recolor(predicts_path, out_path)



