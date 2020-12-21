from deeplab_v3 import Deeplab_v3
from data_utils import DataSet


import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pandas as pd
import numpy as np
from color_utils import color_predicts
from predicts_utils import total_image_predict
from predicts_utils import test_images_predict

from metric_utils import iou

ds='zurich/'

class args:
    batch_size = 16
    lr = 2e-4
    test_display = 100
    weight_decay = 5e-4
    model_name = '1221nir'
    batch_norm_decay = 0.95
    multi_scale = True # 是否多尺度预测
    gpu_num = 1
    pretraining = False
    save_model = 'model/' + ds + model_name
    start_step = 1
    total_steps = 50000

if __name__ == '__main__':
    # 打印以下超参数
    for key in args.__dict__:
        if key.find('__') == -1:
            offset = 20 - key.__len__()
            print(key + ' ' * offset, args.__dict__[key])

    # 使用那一块显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    data_home = '../../data/' + ds + '/'
    data_path_df = pd.read_csv(data_home + 'path_list.csv')
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

    # 更新梯度
    apply_gradient_op = optimizer.apply_gradients(grads_and_vars=grads,
                                                  global_step=tf.train.get_or_create_global_step())
    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    train_op = tf.group(apply_gradient_op, batch_norm_updates_op)

    saver = tf.train.Saver(tf.all_variables())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        graph = tf.get_default_graph()

        if args.pretraining:
            saver = tf.train.import_meta_graph('model/' + args.model_name + '-40000.meta')
            saver.restore(sess, 'model/' + args.model_name + '-40000')
            logits_prob = tf.get_default_graph().get_tensor_by_name("logits_prob:0")
            is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")

        log_path = 'logs/%s/' % args.model_name
        model_path = 'ckpts/%s/' % args.model_name

        if not os.path.exists(model_path): os.makedirs(model_path)
        if not os.path.exists('./logs'): os.makedirs('./logs')
        if not os.path.exists(log_path): os.makedirs(log_path)

        summary_writer = tf.summary.FileWriter('%s/' % log_path, sess.graph)

        learning_rate = args.lr
        for step in range(args.start_step, args.total_steps + 1):
            if step == 30000 or step == 40000:
                learning_rate = learning_rate / 10
            x_tr, y_tr = dataset.next_batch(args.batch_size)

            loss_tr, l2_loss_tr, predicts_tr, _ = sess.run(
                fetches=[cross_entropy, l2_loss, predicts, train_op],
                feed_dict={
                    image: x_tr,
                    label: y_tr,
                    model._is_training: True,
                    lr: learning_rate})

            if step % args.test_display == 0:
                print('current step: ', step, ' loss: ', [loss_tr, l2_loss_tr])

                step_iou = iou(y_pre=np.reshape(predicts_tr, -1),
                               y_true=np.reshape(y_tr, -1))
                print("======================%d======================" % step)
                for key in step_iou.keys():
                    offset = 40 - key.__len__()
                    print(key + ' ' * offset + '%.4f' % step_iou[key])
                test_summary = tf.Summary(
                    value=[tf.Summary.Value(tag=key, simple_value=step_iou[key]) for key in step_iou.keys()]
                )
                # 记录summary
                summary_writer.add_summary(test_summary, step)
                summary_writer.flush()

            if step == 40000 or step == 50000:
                saver.save(sess, args.save_model, global_step=step)
