from network.deeplab_v3_plus import Deeplab_v3
from util.data_utils import DataSet

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import pandas as pd
from util.predicts_utils import *

from util.metric_utils import iou

class args:
    batch_size = 16
    lr = 2e-4
    test_display = 100
    N_GPUS = 4
    weight_decay = 5e-4
    model_name = 'deeplab_4GPU_1211'
    batch_norm_decay = 0.95
    multi_scale = True # 是否多尺度预测
    pretraining = False
    save_model = 'network/' + model_name
    start_step = 1
    total_steps = 50000
    predict = False

def average_gradients(tower_grads):
    avg_grads = []
    # grad_and_vars代表不同的参数（含全部gpu），如四个gpu上对应w1的所有梯度值
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:  # 这里循环的是不同gpu
            expanded_g = tf.expand_dims(g, 0)  # 扩展一个维度代表gpu，如w1=shape(5,10), 扩展后变为shape(1,5,10)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)  # 在第一个维度上合并
        grad = tf.reduce_mean(grad, 0)  # 求平均

        v = grad_and_vars[0][1]  # v 是变量
        grad_and_var = (grad, v)  # 这里是将平均梯度和变量对应起来
        # 将不同变量的平均梯度append一起
        avg_grads.append(grad_and_var)
    # return average gradients
    return avg_grads

# 打印以下超参数
for key in args.__dict__:
    if key.find('__') == -1:
        offset = 20 - key.__len__()
        print(key + ' ' * offset, args.__dict__[key])

data_home = '~/tanrui/deeplabv3-Tensorflow/Test4/dataset/origin/train_origin/'
data_path_df = pd.read_csv(data_home + 'path_list.csv')
data_path_df = data_path_df.sample(frac=1) # 第一次打乱

dataset = DataSet(image_path=data_path_df['image'].values, label_path=data_path_df['label'].values)

model = Deeplab_v3(batch_norm_decay=args.batch_norm_decay)
# network = Unet(batch_norm_decay=args.batch_norm_decay)

all_grads = []
lr = tf.placeholder(tf.float32, )
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
image = tf.placeholder(tf.float32, [None, 256, 256, 4], name='input_x')
label = tf.placeholder(tf.int32, [None, 256, 256])
small_batch_size = int(args.batch_size/args.N_GPUS)
all_predicts = None
with tf.variable_scope(tf.get_variable_scope()):
    for i in tqdm(range(args.N_GPUS)):
        with tf.device('/gpu:%d' % i):
            start = i * small_batch_size
            end = (i + 1) * small_batch_size
            x = image[start:end, :, :, :]
            y = label[start:end, :, :]
            logits = model.forward_pass(x)
            logits_prob = tf.nn.softmax(logits=logits, name='logits_prob' + str(i))
            predicts = tf.argmax(logits, axis=-1, name='predicts' + str(i))
            if all_predicts is None:
                all_predicts = predicts
            else:
                all_predicts = tf.concat((all_predicts, predicts), axis=0)
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
            with tf.name_scope('weight_decay' + str(i)):
                l2_loss = args.weight_decay * tf.add_n(
                    [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
            loss = cross_entropy + l2_loss
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            grads = optimizer.compute_gradients(loss=loss, var_list=tf.trainable_variables())
            for i, (grad, var) in enumerate(grads):
                if grad is not None:
                    grads[i] = (tf.clip_by_norm(grad, 5), var)
            all_grads.append(grads)
            tf.get_variable_scope().reuse_variables()

grads = average_gradients(all_grads)
apply_gradient_op = optimizer.apply_gradients(grads_and_vars=grads,
                                              global_step=tf.train.get_or_create_global_step())
batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
train_op = tf.group(apply_gradient_op, batch_norm_updates_op)
saver = tf.train.Saver(tf.all_variables())

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config) as sess:
    init.run()
    if args.pretraining:
        saver = tf.train.import_meta_graph('network/'+args.model_name+'-40000.meta')
        saver.restore(sess, 'network/'+args.model_name+'-40000')
        logits_prob = tf.get_default_graph().get_tensor_by_name("logits_prob:0")
        is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")

    elif args.predict:
        saver = tf.train.import_meta_graph('network/'+args.model_name+'-50000.meta')
        saver.restore(sess, 'network/'+args.model_name+'-50000')
        all_predicts = None
        for i in range(args.N_GPUS):
            predict = tf.get_default_graph().get_tensor_by_name('predicts' + str(i) + ":0")
            if all_predicts is None:
                all_predicts = predict
            else:
                all_predicts = tf.concat((all_predicts, predict), axis=0)
        is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")
        origin_path = '~/tanrui/deeplabv3-Tensorflow/Test4/dataset/origin/test_origin/'
        images = os.listdir(origin_path + 'images/')
        for i in tqdm(images):
            result = total_image_predict_multigpu(
                origin_path + 'images/' + i,
                image,
                is_training,
                all_predicts,
                sess,
                args.multi_scale
            )
            cv2.imwrite(origin_path + 'results/' + i + '.png', result)

    else:
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
                fetches=[cross_entropy, l2_loss, all_predicts, train_op],
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

            if step % 10000 == 0:
                saver.save(sess, args.save_model, global_step=step)