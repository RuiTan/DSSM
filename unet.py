import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.training import moving_averages

class Unet():
    def __init__(self,
                 batch_norm_decay=0.99,
                 batch_norm_epsilon=1e-3,
                 ):
        self._is_training = tf.placeholder(tf.bool, name='is_training')
        self.num_class = 9
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon

    def forward_pass(self, x):
        with tf.variable_scope('down_sampling', reuse=tf.AUTO_REUSE):
            x = tf.layers.batch_normalization(x, training=self._is_training)
            conv1 = self._conv(tf.expand_dims(x[:,:,:,0], 3), 3, 64, 1, 'conv1_1', True, True)
            conv1 = self._conv(tf.expand_dims(conv1[:,:,:,0], 3), 3, 64, 1, 'conv1_2', True, True)
            pool1 = self._max_pool(conv1, 2, 2, 'pool_1')
            conv2 = self._conv(tf.expand_dims(pool1[:, :, :, 0], 3), 3, 128, 1, 'conv2_1', True, True)
            conv2 = self._conv(tf.expand_dims(conv2[:, :, :, 0], 3), 3, 128, 1, 'conv2_2', True, True)
            pool2 = self._max_pool(conv2, 2, 2, 'pool_2')
            conv3 = self._conv(tf.expand_dims(pool2[:, :, :, 0], 3), 3, 256, 1, 'conv3_1', True, True)
            conv3 = self._conv(tf.expand_dims(conv3[:, :, :, 0], 3), 3, 256, 1, 'conv3_2', True, True)
            pool3 = self._max_pool(conv3, 2, 2, 'pool_3')
            conv4 = self._conv(tf.expand_dims(pool3[:, :, :, 0], 3), 3, 512, 1, 'conv4_1', True, True)
            conv4 = self._conv(tf.expand_dims(conv4[:, :, :, 0], 3), 3, 512, 1, 'conv4_2', True, True)
            drop4 = tf.nn.dropout(conv4, rate=0.5, )
            pool4 = self._max_pool(drop4, 2, 2, 'pool_4')

            conv5 = self._conv(tf.expand_dims(pool4[:, :, :, 0], 3), 3, 1024, 1, 'conv5_1', True, True)
            conv5 = self._conv(tf.expand_dims(conv5[:, :, :, 0], 3), 3, 1024, 1, 'conv5_2', True, True)
            drop5 = tf.nn.dropout(conv5, rate=0.5, )

            up6 = tf.image.resize_bilinear(drop5, tf.shape(drop5)[1:3]*2, name='up6')
            up6 = self._conv(tf.expand_dims(up6[:,:,:,0], 3), 2, 512, 1, 'up6_conv', True, True)
            merge6 = tf.concat([drop4, up6], axis=3, name='merge6')
            conv6 = self._conv(tf.expand_dims(merge6[:, :, :, 0], 3), 3, 512, 1, 'conv6_1', True, True)
            conv6 = self._conv(tf.expand_dims(conv6[:, :, :, 0], 3), 3, 512, 1, 'conv6_2', True, True)

            up7 = tf.image.resize_bilinear(conv6, tf.shape(conv6)[1:3]*2, name='up7')
            up7 = self._conv(tf.expand_dims(up7[:,:,:,0], 3), 2, 256, 1, 'up7_conv', True, True)
            merge7 = tf.concat([conv3, up7], axis=3, name='merge7')
            conv7 = self._conv(tf.expand_dims(merge7[:, :, :, 0], 3), 3, 256, 1, 'conv7_1', True, True)
            conv7 = self._conv(tf.expand_dims(conv7[:, :, :, 0], 3), 3, 256, 1, 'conv7_2', True, True)

            up8 = tf.image.resize_bilinear(conv7, tf.shape(conv7)[1:3] * 2, name='up8')
            up8 = self._conv(tf.expand_dims(up8[:, :, :, 0], 3), 2, 128, 1, 'up8_conv', True, True)
            merge8 = tf.concat([conv2, up8], axis=3, name='merge8')
            conv8 = self._conv(tf.expand_dims(merge8[:, :, :, 0], 3), 3, 128, 1, 'conv8_1', True, True)
            conv8 = self._conv(tf.expand_dims(conv8[:, :, :, 0], 3), 3, 128, 1, 'conv8_2', True, True)

            up9 = tf.image.resize_bilinear(conv8, tf.shape(conv8)[1:3] * 2, name='up9')
            up9 = self._conv(tf.expand_dims(up9[:, :, :, 0], 3), 2, 64, 1, 'up9_conv', True, True)
            merge9 = tf.concat([conv1, up9], axis=3, name='merge9')
            conv9 = self._conv(tf.expand_dims(merge9[:, :, :, 0], 3), 3, 64, 1, 'conv9_1', True, True)
            conv9 = self._conv(tf.expand_dims(conv9[:, :, :, 0], 3), 3, 64, 1, 'conv9_2', True, True)

            conv10 = self._conv(tf.expand_dims(conv9[:, :, :, 0], 3), 1, 9, 1, 'conv10', True, False)
            output = tf.nn.softmax(conv10)
            return output

    def _conv(self,
              x,
              kernel_size,
              filters,
              strides,
              scope,
              batch_norm=False,
              activation=False,
              rate=None
              ):
        """Convolution."""
        with tf.variable_scope(scope):
            x_shape = x.get_shape().as_list()
            w = tf.get_variable(name='weights',
                                shape=[kernel_size, kernel_size, x_shape[3], filters])
            if rate == None:
                x = tf.nn.conv2d(input=x,
                                 filter=w,
                                 padding='SAME',
                                 strides=[1, strides, strides, 1],
                                 name='conv', )
            else:
                x = tf.nn.atrous_conv2d(value=x,
                                        filters=w,
                                        padding='SAME',
                                        name='conv',
                                        rate=rate)

            if batch_norm:
                with tf.variable_scope('BatchNorm'):
                    x = self._batch_norm(x)
            else:
                b = tf.get_variable(name='biases', shape=[filters])
                x = x + b
            if activation:
                x = tf.nn.relu(x)
            return x

    def _batch_norm(self, x):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]

        axis = list(range(len(x_shape) - 1))
        beta = tf.get_variable(name='beta',
                               shape=params_shape,
                               initializer=tf.zeros_initializer)

        gamma = tf.get_variable(name='gamma',
                                shape=params_shape,
                                initializer=tf.ones_initializer)

        moving_mean = tf.get_variable(name='moving_mean',
                                      shape=params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)

        moving_variance = tf.get_variable(name='moving_variance',
                                          shape=params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        tf.add_to_collection('BN_MEAN_VARIANCE', moving_mean)
        tf.add_to_collection('BN_MEAN_VARIANCE', moving_variance)

        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                   mean,
                                                                   self._batch_norm_decay,
                                                                   name='MovingAvgMean')
        update_moving_variance = moving_averages.assign_moving_average(moving_variance,
                                                                       variance,
                                                                       self._batch_norm_decay,
                                                                       name='MovingAvgVariance')

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

        mean, variance = tf.cond(
            pred=self._is_training,
            true_fn=lambda: (mean, variance),
            false_fn=lambda: (moving_mean, moving_variance)
        )
        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, self._batch_norm_epsilon)
        return x

    def _relu(self, x):
        return tf.nn.relu(x)

    def _max_pool(self, x, pool_size, stride, scope):
        with tf.name_scope('max_pool') as name_scope:
            x = tf.layers.max_pooling2d(
                x, pool_size, stride, 'SAME', name=scope
            )
        return x

    def _avg_pool(self, x, pool_size, stride):
        with tf.name_scope('avg_pool') as name_scope:
            x = tf.layers.average_pooling2d(
                x, pool_size, stride, 'SAME')
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _global_avg_pool(self, x):
        with tf.name_scope('global_avg_pool') as name_scope:
            assert x.get_shape().ndims == 4

            x = tf.reduce_mean(x, [1, 2])
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _concat(self, x, y):
        with tf.name_scope('concat') as name_scope:
            assert x.get_shape().ndims == 4
            assert y.get_shape().ndims == 4

            x = tf.concat([x, y], 3)
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _subsample(self, inputs, stride, scope=None):
        """Subsamples the input along the spatial dimensions."""
        if stride == 1:
            return inputs
        else:
            return self._max_pool(inputs, 3, stride, scope)