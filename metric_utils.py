import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import os
# 其他，白色，0
# 植被，绿色，1
# 道路，黑色，2
# 建筑，黄色，3
# 水体，蓝色，4

def iou(y_pre: np.ndarray, y_true: np.ndarray) -> 'dict':
    # print(y_pre)
    # print(y_true)
    # cm是混淆矩阵
    cm = confusion_matrix(
        y_true=y_true,
        y_pred=y_pre,
        # labels=[0, 1, 2, 3, 4]
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    )

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

    return metric_dict


def result():
    origin_path = '/home/gpu/tanrui/deeplabv3-Tensorflow/Test4/dataset/origin/test_origin/'
    labels = []
    predicts = []
    for i in range(1, 16):
        labels.append(str(i) + '.png')
        predicts.append('zh' + str(i) + '.tif.png')
    y_true = None
    y_predict = None
    for i in labels:
        label = cv2.imread(origin_path + 'labels/' + i, cv2.CAP_OPENNI_GRAY_IMAGE)
        if y_true is None:
            y_true = np.reshape(label, -1)
        else:
            y_true = np.append(y_true, label)
    for i in predicts:
        predict = cv2.imread(origin_path + 'results/' + i, cv2.CAP_OPENNI_GRAY_IMAGE)
        if y_predict is None:
            y_predict = np.reshape(predict, -1)
        else:
            y_predict = np.append(y_predict, predict)
    result = iou(y_predict, y_true)
    for key in result.keys():
        offset = 40 - key.__len__()
        print(key + ' ' * offset + '%.4f' % result[key])