import cv2
import numpy as np


def iou_analysis(result):
    ious_with_crop = []
    ious_without_crop = []
    for i in range(result.shape[0]):
        if i % 2 == 0:
            ious_with_crop.append(result[i])
        else:
            ious_without_crop.append(result[i])

    iou_with_crop = ious_with_crop[0]
    iou_without_crop = ious_without_crop[0]
    for i in range(len(ious_with_crop) - 1):
        for j in range(len(ious_with_crop[0])):
            iou_with_crop[j] += ious_with_crop[i + 1][j]
            if iou_with_crop[j] != 0 and ious_with_crop[i + 1][j] != 0:
                iou_with_crop[j] = iou_with_crop[j] / 2

            iou_without_crop[j] += ious_without_crop[i + 1][j]
            if iou_without_crop[j] != 0 and ious_without_crop[i + 1][j] != 0:
                iou_without_crop[j] = iou_without_crop[j] / 2

    return [iou_with_crop, iou_without_crop]


label = cv2.imread('/home/gpu/tanrui/liebling_classify/Fused_result.png', cv2.IMREAD_GRAYSCALE)
labels = [10,32,64,96,128,160,192,224,255]
# labels = [
#     [255, 255, 255]  # 白-Null
#     , [0, 0, 0]  # 黑-Road
#     , [150, 0, 0]  # 红-Bare soil
#     , [0, 125, 0]  # 绿-Tree
#     , [0, 80, 150]  # 蓝-Water
#     , [100, 100, 100]  # 灰-Buildings
#     , [0, 255, 0]  # 青-Grass
#     , [255, 150, 150]  # 黄-Rails
#     , [0, 255, 255]  # 天蓝-Pool
# ]
for i in range(len(labels)):
    label[label == i] = labels[i]
src = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
# src_gray = label
# # RGB在opencv中存储为BGR的顺序,数据结构为一个3D的numpy.array,索引的顺序是行,列,通道:
# B = src[:,:,0]
# G = src[:,:,1]
# R = src[:,:,2]
# R = 255 - R
# # 灰度g=p*R+q*G+t*B（其中p=0.2989,q=0.5870,t=0.1140），于是B=(g-p*R-q*G)/t。于是我们只要保留R和G两个颜色分量，再加上灰度图g，就可以回复原来的RGB图像。
# p = 0.2989; q = 0.5870; t = 0.1140
# G_new = (G-p*R-t*B)/q
# G_new = np.uint8(G_new)
# src_new = np.zeros((src.shape)).astype("uint8")
# src_new[:,:,0] = B
# src_new[:,:,1] = G_new
# src_new[:,:,2] = R
cv2.imwrite('/home/gpu/tanrui/liebling_classify/Fused_result2.png',src)
