import os

import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from util.metric_utils import iou
from util.predicts_utils import multi_scale_predict
from util.color_utils import color_predicts
import numpy as np
from tqdm import tqdm
from util.predicts_utils import read_tiff_img

ds = 'potsdam/'
model_path = '1222_nir-50000'

class args:
    batch_size = 16
    lr = 2e-4
    test_display = 500
    weight_decay = 5e-4
    model_name = ds + model_path
    batch_norm_decay = 0.95
    multi_scale = False  # 是否多尺度预测
    gpu_num = 1
    pretraining = True
    origin_image_path = '../../data/' + ds + 'images/'
    origin_label_path = '../../data/' + ds + 'labels/'

os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu_num

class Crop:
    def __init__(self, crop: np.ndarray, x_start: int, x_end: int, y_start: int, y_end: int):
        self.crop = crop
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end

    def predict_crop(self, input_placeholder: tf.placeholder,
                     is_training_placeholder: tf.placeholder, logits_prob_node: tf.Tensor,
                     sess: tf.Session, prob: bool) -> np.ndarray:
        predict_result = multi_scale_predict(self.crop, input_placeholder, is_training_placeholder, logits_prob_node,
                                             sess, prob)
        return predict_result[self.y_start: self.y_end, self.x_start:self.x_end]


def scale(image: np.ndarray, rate: int, channel=4):
    width = image.shape[0]
    height = image.shape[1]
    w_step = width // rate
    h_step = height // rate
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


def crops_predict(source_image_path: str,
                  input_placeholder: tf.placeholder,
                  is_training_placeholder: tf.placeholder,
                  logits_prob_node: tf.Tensor,
                  sess: tf.Session,
                  multi_scale=False
                  ):
    # source_image = cv2.imread(source_image_path)
    # source_image = TIFF.open(source_image_path, mode='r').read_image()
    source_image,_ = read_tiff_img(source_image_path)
    print(source_image.shape)
    # source_image = tiff_change_color(source_image)
    height = source_image.shape[0]
    width = source_image.shape[1]
    h_step = height // 128
    w_step = width // 128
    h_rest = height - h_step * 128
    w_rest = width - w_step * 128

    source_image_predict = []
    print(h_step*w_step, ' steps needed')
    for h in tqdm(range(h_step)):
        row_predict = []
        for w in range(w_step):
            # overlap切片
            if h == h_step - 1 and w == w_step - 1:
                crop = source_image[height - 256: height, width - 256: width, :]
            elif h == h_step - 1:
                crop = source_image[height - 256: height, w * 128: w * 128 + 256, :]
            elif w == w_step - 1:
                crop = source_image[h * 128: h * 128 + 256, width - 256: width, :]
            else:
                crop = source_image[h * 128: h * 128 + 256, w * 128: w * 128 + 256, :]
            # 设置有效识别区域
            x_start, x_end, y_start, y_end = 64, 192, 64, 192
            if h == 0:
                y_start = 0
            if w == 0:
                x_start = 0
            if h == h_step - 1:
                y_start = 192 - h_rest
                y_end = 256
            if w == w_step - 1:
                x_start = 192 - w_rest
                x_end = 256
            crop_predict = Crop(crop=crop, x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end) \
                .predict_crop(input_placeholder, is_training_placeholder, logits_prob_node, sess, multi_scale)
            if w == 0:
                row_predict = crop_predict
            else:
                row_predict = np.hstack((row_predict, crop_predict))
        if h == 0:
            source_image_predict = row_predict
        else:
            source_image_predict = np.vstack((source_image_predict, row_predict))
    return source_image_predict


def write_test_result(filename: str, img: str, label: str, predict_func, input_placeholder: tf.placeholder,
                         is_training_placeholder: tf.placeholder, logits_prob_node: tf.Tensor,
                         sess: tf.Session, multi_scale=False):
    source_predict, labels = predict_func(
        img,
        label,
        input_placeholder=input_placeholder,
        logits_prob_node=logits_prob_node,
        is_training_placeholder=is_training_placeholder,
        sess=sess,
        multi_scale=multi_scale
    )
    cv2.imwrite(filename=filename, img=color_predicts(img=source_predict))

    result = iou(y_pre=np.reshape(source_predict, -1),
                 y_true=np.reshape(labels, -1))
    return list(result.values())


def tiff_change_color(data, scale=255, data_type=np.uint8):
    """
    使用线性拉伸，对uint16格式的图像进行色彩转换
    :param data: 图像的numpy array
    :return: 处理后的图像数据，格式不变
    """
    # max_value = 1000
    max_value = data.max()
    min_value = data.min()
    print('min_val is '+str(min_value)+', max_val is '+str(max_value))
    result = ((data-min_value)/(max_value-min_value)*scale).astype(data_type)
    return result

def write_predict_result(filename: str, img: str, label: str, predict_func, input_placeholder: tf.placeholder,
                         is_training_placeholder: tf.placeholder, logits_prob_node: tf.Tensor,
                         sess: tf.Session, multi_scale=False):
    source_predict = predict_func(
        img,
        input_placeholder=input_placeholder,
        logits_prob_node=logits_prob_node,
        is_training_placeholder=is_training_placeholder,
        sess=sess,
        multi_scale=multi_scale
    )
    cv2.imwrite(filename=filename, img=color_predicts(img=source_predict))

    # labels = cv2.imread(label, cv2.CAP_OPENNI_GRAY_IMAGE)
    # result = iou(y_pre=np.reshape(source_predict, -1),
    #              y_true=np.reshape(labels, -1))
    # return list(result.values())


saver = tf.train.import_meta_graph('network/' + args.model_name + '.meta')
# predict_prefix = 'potsdam_scale_0810/'
predict_path = '../../data/' + ds + 'result/' + model_path + '/'
if not os.path.exists(predict_path): os.makedirs(predict_path)

with tf.Session() as sess:
    saver.restore(sess, 'network/' + args.model_name + '')

    logits_prob = tf.get_default_graph().get_tensor_by_name("logits_prob:0")
    is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")
    image = tf.get_default_graph().get_tensor_by_name("input_x:0")

    '''
    对所有图像重叠或循环切片预测结果，并拼接起来，最后计算mIOU
    '''
    write_images = os.listdir(args.origin_image_path)

    # write_images = ['top_potsdam_5_10_RGBIR.tif','top_potsdam_5_11_RGBIR.tif','top_potsdam_5_12_RGBIR.tif',
    #                 'top_potsdam_5_13_RGBIR.tif','top_potsdam_5_14_RGBIR.tif','top_potsdam_5_15_RGBIR.tif']
    # write_images = ['Fused.tiff','Origin.tiff','Processed.tiff']
    # write_images = ['PNN_up_img1.tiff','1_MS_cut.tiff']
    # total_iou = []
    for i in tqdm(write_images):
        write_predict_result(filename=predict_path + model_path.split('-')[0] + i + '.png',
                             img=args.origin_image_path + i,
                             label=args.origin_label_path + str(i).replace('RGBIR.tif', 'label.png'),
                             predict_func=crops_predict,
                             input_placeholder=image,
                             logits_prob_node=logits_prob,
                             is_training_placeholder=is_training,
                             sess=sess,
                             multi_scale=args.multi_scale
                             )
        # non_crop_result = write_predict_result(filename='%s%s_%d_predict.png' % (predict_path, i, 50000),
        #                            img=args.origin_image_path + i,
        #                            label=args.origin_label_path + str(i).split('.')[0][2:] + '.png',
        #                            predict_func=total_image_predict,
        #                            input_placeholder=image,
        #                            logits_prob_node=logits_prob,
        #                            is_training_placeholder=is_training,
        #                            sess=sess,
        #                            multi_scale=args.multi_scale
        #                            )

    #     if len(total_iou):
    #         total_iou = np.vstack((total_iou, np.vstack((crop_result, non_crop_result))))
    #     else:
    #         total_iou = np.vstack((crop_result, non_crop_result))
    #
    # result_iou = iou_analysis(total_iou)
    # print(result_iou)
    # result = open('predict/result.txt', mode='w+')
    # result.write(str(datetime.datetime.now()) + ': ' + str(result_iou))
    # result.write('\n')
    # result.close()

    #     '''
    #         1000张测试图片，每次随机取200张，共取10次，结果计算平均IOU
    #     '''
    #     mean_iou_result = []
    #     for i in tqdm(range(10)):
    #         test_predict, test_label = test_images_predict(
    #             ori_image_path=args.test_image_path,
    #             ori_label_path=args.test_label_path,
    #             test_size=200,
    #             input_placeholder=image,
    #             logits_prob_node=logits_prob,
    #             is_training_placeholder=is_training,
    #             sess=sess,
    #             multi_scale=args.multi_scale
    #         )
    #         test_result = iou(y_pre=np.reshape(test_predict, -1), y_true=np.reshape(test_label, -1))
    #         mean_iou_result.append(list(test_result.values()))
    #
    #     result = mean_iou_result[0]
    #     for i in range(len(mean_iou_result) - 1):
    #         for j in range(len(mean_iou_result[0])):
    #             result[j] += mean_iou_result[i + 1][j]
    #             if result[j] != 0 and mean_iou_result[i + 1][j] != 0:
    #                 result[j] = result[j] / 2
    #     print(result)
    #
    #         # for key in test_result.keys():
    #         #     offset = 40 - key.__len__()
    #         #     print(key + ' ' * offset + '%.4f' % test_result[key])
    #