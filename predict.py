from deeplab_v3_plus_weighted import Deeplab_v3
from data_utils import DataSet

import cv2
import os
import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
from color_utils import color_predicts
from predicts_utils import total_image_predict
from predicts_utils import test_images_predict
from result import crops_predict

from metric_utils import iou
from tqdm import tqdm


class args:
    batch_size = 16
    lr = 2e-4
    test_display = 1000
    weight_decay = 5e-4
    model_name = 'deeplab_v3_plus_20200809-40000'
    batch_norm_decay = 0.95
    test_image_path = 'dataset/test/images/'
    test_label_path = 'dataset/test/labels/'
    multi_scale = False  # 是否多尺度预测
    gpu_num = 1
    pretraining = True
    origin_image_path = 'dataset/origin/images/'
    save_model = 'model/' + model_name
    start_step = 1
    total_steps = 50000


# 打印以下超参数
for key in args.__dict__:
    if key.find('__') == -1:
        offset = 20 - key.__len__()
        print(key + ' ' * offset, args.__dict__[key])

# 使用那一块显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu_num

data_path_df = pd.read_csv('dataset/path_list.csv')
data_path_df = data_path_df.sample(frac=1)  # 第一次打乱

dataset = DataSet(image_path=data_path_df['image'].values, label_path=data_path_df['label'].values)

model = Deeplab_v3(batch_norm_decay=args.batch_norm_decay)

image = tf.placeholder(tf.float32, [None, 256, 256, 4], name='input_x')
label = tf.placeholder(tf.int32, [None, 256, 256])
lr = tf.placeholder(tf.float32, )

logits = model.forward_pass(image)
logits_prob = tf.nn.softmax(logits=logits, name='logits_prob')
predicts = tf.argmax(logits, axis=-1, name='predicts')

# variables_to_restore = tf.trainable_variables(scope='resnet_v2_50')

# finetune resnet_v2_50的参数(block1到block4)
# restorer = tf.train.Saver(variables_to_restore)
# cross_entropy
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label))

# 只将weight加入到weight_decay中
# https://arxiv.org/pdf/1807.11205.pdf
# weight_for_weightdecay = []
# for var in tf.trainable_variables():
#     if var.name.__contains__('weight'):
#         weight_for_weightdecay.append(var)
#         print(var.op.name)
#     else:
#         continue
# l2_norm l2正则化
with tf.name_scope('weight_decay'):
    l2_loss = args.weight_decay * tf.add_n(
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])

optimizer = tf.train.AdamOptimizer(learning_rate=lr)
loss = cross_entropy + l2_loss

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# 计算梯度
grads = optimizer.compute_gradients(loss=loss, var_list=tf.trainable_variables())
# for grad, var in grads:
#     if grad is not None:
#         tf.summary.histogram(name='%s_gradients' % var.op.name, values=grad)
#         tf.summary.histogram(name='%s' % var.op.name, values=var)
# 梯度裁剪
# gradients, variables = zip(*grads)
# gradients, global_norm = tf.clip_by_global_norm(gradients, 5)

# 更新梯度
apply_gradient_op = optimizer.apply_gradients(grads_and_vars=grads, global_step=tf.train.get_or_create_global_step())
batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
train_op = tf.group(apply_gradient_op, batch_norm_updates_op)

saver = tf.train.Saver(tf.all_variables())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# summary_op = tf.summary.merge_all()

with tf.Session(config=config) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()

    if args.pretraining:
        # finetune resnet_v2_50参数，需要下载权重文件
        # restorer.restore(sess, 'ckpts/resnet_v2_50/resnet_v2_50.ckpt')
        saver = tf.train.import_meta_graph('model/'+args.model_name+'.meta')
        saver.restore(sess, 'model/'+args.model_name)
        logits_prob = tf.get_default_graph().get_tensor_by_name("logits_prob:0")
        is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")

        # origin_path = '/home/gpu/tanrui/liebling_classify/'
        # origin_path = '/home/gpu/tanrui/tanrui/data/ftp.ipi.uni-hannover.de/ISPRS_BENCHMARK_DATASETS/Potsdam/4_Ortho_RGBIR/'
        origin_path = '/home/gpu/tanrui/tanrui/deeplabv3-Tensorflow/Test4/dataset/origin/test_origin/images/'
        images = ['zh1']
        for i in tqdm(images):
            result = total_image_predict(
                origin_path + i + '.tif',
                image,
                is_training,
                logits_prob,
                sess,
                args.multi_scale
            )
            cv2.imwrite(origin_path+'result/'+i+'.png',result)
            # labels = [10, 32, 64, 96, 128, 160, 192, 224, 255]
            # label = result
            # for j in range(len(labels)):
            #     label[label == j] = labels[j]
            # cv2.imwrite(origin_path+'result/'+i+'_2.png', label)