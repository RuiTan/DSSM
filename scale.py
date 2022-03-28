import numpy as np
from util.predicts_utils import read_tiff_img
from tqdm import tqdm
import cv2

def scale(image: np.ndarray, rate: int, channel=4):
    width = image.shape[0]
    height = image.shape[1]
    w_step = width // rate
    h_step = height // rate
    if channel == 1:
        result = np.zeros((w_step, h_step), dtype=np.uint8)
        for w in range(w_step):
            for h in range(h_step):
                start_x = w * rate
                start_y = h * rate
                end_x = start_x + rate
                end_y = start_y + rate
                result[w, h] = np.mean(image[start_x:end_x, start_y:end_y])
    else:
        result = np.zeros((w_step, h_step, channel), dtype=np.uint8)
        for i in range(channel):
            for w in range(w_step):
                for h in range(h_step):
                    start_x = w * rate
                    start_y = h * rate
                    end_x = start_x + rate
                    end_y = start_y + rate
                    result[w, h, i] = np.mean(image[start_x:end_x,start_y:end_y,i])
    return result


origin_image_path = '/home/gpu/tanrui/tanrui/data/ftp.ipi.uni-hannover.de/ISPRS_BENCHMARK_DATASETS/Potsdam/label/'
write_path = 'dataset/scale/potsdam/labels/'
# image_paths = os.listdir(origin_image_path)
image_paths = ['top_potsdam_4_10_label.tif']
for image_path in tqdm(image_paths):
    if image_path.endswith('.tif'):
        image, _ = read_tiff_img(origin_image_path + image_path)
        # image = cv2.imread(origin_image_path + image_path, cv2.CAP_OPENNI_GRAY_IMAGE)
        # labels = [255, 150, 76, 226, 29, 179]
        # for l in range(len(labels)):
        #     image[image == labels[l]] = l
        image = scale(image, rate=8, channel=3)
        cv2.imwrite(write_path+image_path.split(".")[0]+'.png', image)
        # write_tiff(write_path + image_path, image)
