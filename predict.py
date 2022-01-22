from network.deeplab_v3_plus_weighted import Deeplab_v3

import cv2
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from util.predicts_utils import total_image_predict

from tqdm import tqdm

ds = 'zurich/'
model_path = ds + '1216_rgbnir-50000'

class args:
    batch_size = 16
    lr = 2e-4
    test_display = 1000
    weight_decay = 5e-4
    model_name = model_path
    batch_norm_decay = 0.95
    multi_scale = False  # 是否多尺度预测
    gpu_num = 1
    pretraining = True


# 打印以下超参数
for key in args.__dict__:
    if key.find('__') == -1:
        offset = 20 - key.__len__()
        print(key + ' ' * offset, args.__dict__[key])

# 使用那一块显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu_num

model = Deeplab_v3(batch_norm_decay=args.batch_norm_decay)

image = tf.placeholder(tf.float32, [None, 256, 256, 4], name='input_x')
label = tf.placeholder(tf.int32, [None, 256, 256])
lr = tf.placeholder(tf.float32, )

logits = model.forward_pass(image)
logits_prob = tf.nn.softmax(logits=logits, name='logits_prob')
# predicts = tf.argmax(logits, axis=-1, name='predicts')
# cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label))

with tf.name_scope('weight_decay'):
    l2_loss = args.weight_decay * tf.add_n(
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
#
# optimizer = tf.train.AdamOptimizer(learning_rate=lr)
# loss = cross_entropy + l2_loss
#
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# # 计算梯度
# grads = optimizer.compute_gradients(loss=loss, var_list=tf.trainable_variables())
#
# # 更新梯度
# apply_gradient_op = optimizer.apply_gradients(grads_and_vars=grads, global_step=tf.train.get_or_create_global_step())
# batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
# train_op = tf.group(apply_gradient_op, batch_norm_updates_op)

saver = tf.train.Saver(tf.all_variables())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()

    if args.pretraining:
        print(args.model_name)
        saver = tf.train.import_meta_graph('network/'+args.model_name+'.meta')
        saver.restore(sess, 'network/'+args.model_name)
        logits_prob = tf.get_default_graph().get_tensor_by_name("logits_prob:0")
        is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")

        origin_path = '../../data/' + ds
        images = os.listdir(origin_path + 'images/')
        for i in tqdm(images):
            result = total_image_predict(
                origin_path + 'images/' + i + '.tif',
                image,
                is_training,
                logits_prob,
                sess,
                args.multi_scale
            )
            cv2.imwrite(origin_path+'predict/'+i+'.png',result)
            # labels = [10, 32, 64, 96, 128, 160, 192, 224, 255]
            # label = result
            # for j in range(len(labels)):
            #     label[label == j] = labels[j]
            # cv2.imwrite(origin_path+'result/'+i+'_2.png', label)