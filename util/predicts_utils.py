import os
import random

import cv2
import numpy as np
import rasterio
import tensorflow.compat.v1 as tf
from libtiff import TIFF
from tqdm import tqdm

size = 256
batch = 64
test_images_count = 1000

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

def predict(image: np.ndarray, input_placeholder: tf.placeholder,
    is_training_placeholder: tf.placeholder, logits_prob_node: tf.Tensor,
    sess: tf.Session, prob: bool) -> np.ndarray:
    """
    使用模型预测一张图片，并输出其预测概率分布
    Args:
        image: np.ndarray [size, size, 3] 需要预测的图片数组
        input_placeholder: tf.placeholder
        is_training_placeholder: tf.placeholder
        logits_prob_node: tf.Tensor [size, size, num_classes]
        sess: tf.Session
        prob: bool 输出的是概率，还是预测值
    Returns:
        image_predict: np.ndarray [size, size, 5] if porb is True
                       np.ndarray [size, size] if prob is not True
    """
    # assert image.shape == (256, 256, 3), print(image.shape)
    # 给image升维 [256, 256, 3] -> [1, 256, 256, 3]
    feed_dict = {input_placeholder: np.expand_dims(image, 0),
                 is_training_placeholder: False}
    image_predict_prob = sess.run(logits_prob_node, feed_dict=feed_dict)
    # 给image降维 [1, 256, 256, 5] -> [256, 256, 5]
    image_predict_prob = np.squeeze(image_predict_prob, 0)
    if prob:
        # 输出预测概率分布
        return image_predict_prob
    else:
        # 输出预测值
        image_predict = np.argmax(image_predict_prob, -1)
        return image_predict

def rotate(x, angle):
    """ 旋转函数
    """
    M_rotate = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    x = cv2.warpAffine(x, M_rotate, (size, size))
    return x

def multi_scale_predict(image: np.ndarray, input_placeholder: tf.placeholder,
    is_training_placeholder: tf.placeholder, logits_prob_node: tf.Tensor,
    sess: tf.Session, multi: bool):
    """
    Args:
        image:
        input_placeholder:
        is_training_placeholder:
        logits_prob_node:
        sess:
        multi:

    Returns:
        np.ndarray [size, size]
    """

    # 旋转函数
    kwargs = {
        'input_placeholder':input_placeholder,
        'is_training_placeholder':is_training_placeholder,
        'logits_prob_node':logits_prob_node,
        'sess':sess,
        'prob':True,
    }
    if multi:
        image_predict_prob_list = [
            predict(image=image, **kwargs)
        ]
        # 旋转三个
        angle_list = [90, 180, 270]
        for angle in angle_list:
            image_rotate = rotate(image, angle)

            image_rotate_predict_prob = predict(image=image_rotate, **kwargs)
            image_predict_prob = rotate(image_rotate_predict_prob, -1 * angle)
            image_predict_prob_list.append(image_predict_prob)
        # 翻转两个
        flip_list = [1, 0]
        for mode in flip_list:
            image_flip = cv2.flip(image, mode)

            image_flip_predict_prob = predict(image=image_flip, **kwargs)
            image_predict_prob = cv2.flip(image_flip_predict_prob, mode)
            image_predict_prob_list.append(image_predict_prob)
        # 求和平均
        final_predict_prob = sum(image_predict_prob_list) / len(image_predict_prob_list)
        return np.argmax(final_predict_prob, -1)
    else:
        kwargs['prob'] = False
        return predict(image, **kwargs)

def total_image_predict_multigpu(ori_image_path: str,
                        input_placeholder: tf.placeholder,
                        is_training_placeholder: tf.placeholder,
                        all_predicts: tf.Tensor,
                        sess: tf.Session,
                        multi_scale = False
                        ) -> np.ndarray:
    # ori_image = cv2.imread(ori_image_path, cv2.CAP_OPENNI_GRAY_IMAGE)
    ori_image, ori_meta = read_tiff_img(ori_image_path)

    # 开始切图 cut
    h_step = ori_image.shape[0] // 256
    w_step = ori_image.shape[1] // 256

    h_rest = -(ori_image.shape[0] - 256 * h_step)
    w_rest = -(ori_image.shape[1] - 256 * w_step)

    image_list = []
    predict_list = []
    # 循环切图
    for h in range(h_step):
        for w in range(w_step):
            # 划窗采样
            image_sample = ori_image[(h * 256):(h * 256 + 256),
                           (w * 256):(w * 256 + 256), :]
            image_list.append(image_sample)
        image_list.append(ori_image[(h * 256):(h * 256 + 256), -256:, :])
    for w in range(w_step - 1):
        image_list.append(ori_image[-256:, (w * 256):(w * 256 + 256), :])
    image_list.append(ori_image[-256:, -256:, :])

    # 对每个图像块预测
    # predict
    batches = int(len(image_list)/batch) + 1
    for i in tqdm(range(batches)):
        image_batch = np.array(image_list[i*batch:(i+1)*batch])
        feed_dict = {input_placeholder: image_batch,
                     is_training_placeholder: False}
        predict = sess.run(all_predicts, feed_dict=feed_dict)
        # 保存覆盖小图片
        # predict_list.append(predict)
        if len(predict_list) == 0:
            predict_list = predict
        else:
            predict_list = np.concatenate((predict_list, predict),axis=0)
    predict_list = np.array(predict_list)
    # 将预测后的图像块再拼接起来
    count_temp = 0
    tmp = np.ones([ori_image.shape[0], ori_image.shape[1]])
    for h in range(h_step):
        for w in range(w_step):
            tmp[
            h * 256:(h + 1) * 256,
            w * 256:(w + 1) * 256
            ] = predict_list[count_temp]
            count_temp += 1
        tmp[h * 256:(h + 1) * 256, w_rest:] = predict_list[count_temp][:, w_rest:]
        count_temp += 1
    for w in range(w_step - 1):
        tmp[h_rest:, (w * 256):(w * 256 + 256)] = predict_list[count_temp][h_rest:, :]
        count_temp += 1
    tmp[h_rest:, w_rest:] = predict_list[count_temp][h_rest:, w_rest:]
    return tmp

def total_image_predict(ori_image_path: str,
                        input_placeholder: tf.placeholder,
                        is_training_placeholder: tf.placeholder,
                        logits_prob_node: tf.Tensor,
                        sess: tf.Session,
                        multi_scale = False
                        ) -> np.ndarray:
    # ori_image = cv2.imread(ori_image_path, cv2.CAP_OPENNI_GRAY_IMAGE)
    ori_image, ori_meta = read_tiff_img(ori_image_path)

    # 开始切图 cut
    h_step = ori_image.shape[0] // 256
    w_step = ori_image.shape[1] // 256

    h_rest = -(ori_image.shape[0] - 256 * h_step)
    w_rest = -(ori_image.shape[1] - 256 * w_step)

    image_list = []
    predict_list = []
    # 循环切图
    for h in range(h_step):
        for w in range(w_step):
            # 划窗采样
            image_sample = ori_image[(h * 256):(h * 256 + 256),
                           (w * 256):(w * 256 + 256), :]
            image_list.append(image_sample)
        image_list.append(ori_image[(h * 256):(h * 256 + 256), -256:, :])
    for w in range(w_step - 1):
        image_list.append(ori_image[-256:, (w * 256):(w * 256 + 256), :])
    image_list.append(ori_image[-256:, -256:, :])

    # 对每个图像块预测
    # predict
    batches = int(len(image_list)/batch) + 1
    for i in tqdm(image_list):
        # image_batch = np.array(image_list[i*batch:(i+1)*batch])
        predict = multi_scale_predict(
            image=i,
            input_placeholder=input_placeholder,
            is_training_placeholder=is_training_placeholder,
            logits_prob_node=logits_prob_node,
            sess=sess,
            multi=multi_scale
        )
        # 保存覆盖小图片
        predict_list.append(predict)
        # if len(predict_list) == 0:
        #     predict_list = predict
        # else:
        #     predict_list = np.concatenate((predict_list, predict),axis=0)
    predict_list = np.array(predict_list)
    # 将预测后的图像块再拼接起来
    count_temp = 0
    tmp = np.ones([ori_image.shape[0], ori_image.shape[1]])
    for h in range(h_step):
        for w in range(w_step):
            tmp[
            h * 256:(h + 1) * 256,
            w * 256:(w + 1) * 256
            ] = predict_list[count_temp]
            count_temp += 1
        tmp[h * 256:(h + 1) * 256, w_rest:] = predict_list[count_temp][:, w_rest:]
        count_temp += 1
    for w in range(w_step - 1):
        tmp[h_rest:, (w * 256):(w * 256 + 256)] = predict_list[count_temp][h_rest:, :]
        count_temp += 1
    tmp[h_rest:, w_rest:] = predict_list[count_temp][h_rest:, w_rest:]
    return tmp

def test_images_predict(ori_image_path: str,
                        ori_label_path: str,
                        test_size: int,
                        input_placeholder: tf.placeholder,
                        is_training_placeholder: tf.placeholder,
                        logits_prob_node: tf.Tensor,
                        sess: tf.Session,
                        multi_scale = False
                        ):
    images = os.listdir(ori_image_path)
    predicts = []
    labels = []
    # test_random = random.sample(range(1, len(images)), 10)

    for i in range(test_size):
        rand_index = random.randint(1, test_images_count - 1)
        # image = cv2.imread(ori_image_path + images[i])
        image = TIFF.open(ori_image_path + images[rand_index]).read_image()
        label = cv2.imread(ori_label_path + images[rand_index].split('.')[0] + '.png', cv2.CAP_OPENNI_GRAY_IMAGE)
        multi_predict = multi_scale_predict(
            image=image,
            input_placeholder=input_placeholder,
            is_training_placeholder=is_training_placeholder,
            logits_prob_node=logits_prob_node,
            sess=sess,
            multi=multi_scale
        )
        if i == 0:
            predicts = multi_predict
            labels = label
        else:
            predicts = np.hstack((predicts, multi_predict))
            labels = np.hstack((labels, label))

    return predicts, labels