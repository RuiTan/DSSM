import numpy as np
import cv2

# 给标签图上色
def color_predicts(img):

    '''
    给class图上色
    '''
    return label_to_color(img)

def color_annotation(label_path, output_path):
    img = cv2.imread(label_path,cv2.CAP_OPENNI_GRAY_IMAGE)
    color = label_to_color(img)
    cv2.imwrite(output_path,color)

def label_to_color(img: np.ndarray) -> np.ndarray:
    color = np.ones([img.shape[0], img.shape[1], 3])
    color[img==0] = [255, 255, 255] #白-Null
    color[img==1] = [0, 0, 0] #黑-Road
    color[img==2] = [150,0,0] #红-Bare soil
    color[img==3] = [0, 125, 0] #绿-Tree
    color[img==4] = [0,80,150] #蓝-Water
    color[img==5] = [100,100,100] #灰-Buildings
    color[img==6] = [0,255 ,0] #青-Grass
    color[img==7] = [255,150,150] #黄-Rails
    color[img==8] = [0,255,255] #天蓝-Pool
    return color

