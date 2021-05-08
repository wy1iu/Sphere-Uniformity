import tensorflow as tf
import numpy as np

class VGG():
    def get_conv_filter(self, shape, reg, stddev):
        # seed_num=shape[0]*shape[1]*shape[2]*shape[3]
        init = tf.random_normal_initializer(stddev=stddev)
        if reg:
            regu = tf.contrib.layers.l2_regularizer(self.wd)
            filt = tf.get_variable('filter', shape, initializer=init,regularizer=regu)
        else:
            filt = tf.get_variable('filter', shape, initializer=init)

        return filt   

    def get_p(self, shape):
        init = tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32)
        regu = tf.contrib.layers.l2_regularizer(self.wd)
        p = tf.get_variable('project', shape, initializer=init,regularizer=regu)
        
        return p    

    def get_bias(self, dim, init_bias, name):
        with tf.variable_scope(name):
            init = tf.constant_initializer(init_bias)
            regu = tf.contrib.layers.l2_regularizer(self.wd)
            bias = tf.get_variable('bias', dim, initializer=init, regularizer=regu)

            return bias

    def cal(self, filt):
        new_filt = tf.reduce_sum(filt,axis=[1])
        new_filt_norm = tf.norm(new_filt)
        return new_filt_norm

    def batch_norm(self, x, n_out, phase_train):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope('bn'):

            gamma = self.get_bias(n_out, 1.0, 'gamma')
            beta = self.get_bias(n_out, 0.0, 'beta')

            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.999)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            return tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME', name=name)

    def _add_thomson_first(self, filt, n_filt):
        filt = tf.reshape(filt, [-1, n_filt])
        loss = 0.01*self.cal(filt)

        tf.add_to_collection('thomson_loss', loss)

    def _add_thomson_final(self, filt, n_filt):
        filt = tf.reshape(filt, [-1, n_filt])
        loss = 0.1*self.cal(filt)

        tf.add_to_collection('thomson_final', loss)

    def _conv_layer(self, bottom, ksize, n_filt, is_training, lr_, name, stride=1, 
        pad='SAME', relu=False, reg=True, thom='False', bn=True):

        with tf.variable_scope(name) as scope:
            n_input = bottom.get_shape().as_list()[3]
            shape = [ksize, ksize, n_input, n_filt]
            print("shape of filter %s: %s" % (name, str(shape)))

            filt = self.get_conv_filter(shape, reg, stddev=tf.sqrt(2.0/tf.to_float(ksize*ksize*n_input)))
            if thom == 'True':
                self._add_thomson_first(filt, n_filt)
            elif thom == 'Final':
                self._add_thomson_final(filt, n_filt)


            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding=pad)

            if bn:
                conv = self.batch_norm(conv, n_filt, is_training)
                
            if relu:
                return tf.nn.relu(conv)
            else:
                return conv

    def build(self, rgb, n_class, is_training, lr_):
        self.wd = 5e-4

        feat = (rgb - 127.5) / 128.0

        ksize = 3
        n_layer = 3

        # 32X32
        n_out = 64
        for i in range(n_layer):
            feat = self._conv_layer(feat, ksize, n_out, is_training, lr_, name="conv1_" + str(i), bn=True, relu=True,
                                    pad='SAME',  reg=True, thom='True')
        feat = self._max_pool(feat, 'pool1')

        # 16X16
        n_out = 128
        for i in range(n_layer):
            feat = self._conv_layer(feat, ksize, n_out, is_training, lr_, name="conv2_" + str(i), bn=True, relu=True,
                                    pad='SAME',  reg=True, thom='True')
        feat = self._max_pool(feat, 'pool2')

        # 8X8
        n_out = 256
        for i in range(n_layer):
            feat = self._conv_layer(feat, ksize, n_out, is_training, lr_, name="conv3_" + str(i), bn=True, relu=True,
                                    pad='SAME',  reg=True, thom='True')
        feat = self._max_pool(feat, 'pool3')

        self.fc6 = self._conv_layer(feat, 4, 256, is_training, lr_, name="fc6", bn=False, relu=False, pad='VALID',
                                    reg=True, thom='True')

        self.score = self._conv_layer(self.fc6, 1, n_class, is_training, lr_, name="score", bn=False, relu=False, pad='VALID',
                                      reg=True,  thom='Final')

        self.pred = tf.squeeze(tf.argmax(self.score, axis=3))
