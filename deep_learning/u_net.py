import tensorflow as tf
import tensorlayer as tl
import tensorlayer.layers as tl_layer
import numpy as np

import tensorflow.keras.layers as tfk_layer
import tensorflow.keras.models as tfk_model

'''08/19/2019 copied from https://github.com/zsdonghao/u-net-brain-tumor'''


def u_net(x, is_train=False, reuse=False, n_out=1):
    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tl_layer.InputLayer(x, name='inputs')
        conv1 = tl_layer.Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
        conv1 = tl_layer.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
        pool1 = tl_layer.MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = tl_layer.Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, name='conv2_1')
        conv2 = tl_layer.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='conv2_2')
        pool2 = tl_layer.MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = tl_layer.Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, name='conv3_1')
        conv3 = tl_layer.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='conv3_2')
        pool3 = tl_layer.MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = tl_layer.Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, name='conv4_1')
        conv4 = tl_layer.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='conv4_2')
        pool4 = tl_layer.MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = tl_layer.Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, name='conv5_1')
        conv5 = tl_layer.Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, name='conv5_2')

        up4 = tl_layer.DeConv2dLayer(conv5, 512, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')
        up4 = tl_layer.ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = tl_layer.Conv2d(up4, 512, (3, 3), act=tf.nn.relu, name='uconv4_1')
        conv4 = tl_layer.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='uconv4_2')
        up3 = tl_layer.DeConv2dLayer(conv4, 256, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3')
        up3 = tl_layer.ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = tl_layer.Conv2d(up3, 256, (3, 3), act=tf.nn.relu, name='uconv3_1')
        conv3 = tl_layer.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='uconv3_2')
        up2 = tl_layer.DeConv2dLayer(conv3, 128, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')
        up2 = tl_layer.ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = tl_layer.Conv2d(up2, 128, (3, 3), act=tf.nn.relu,  name='uconv2_1')
        conv2 = tl_layer.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='uconv2_2')
        up1 = tl_layer.DeConv2dLayer(conv2, 64, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')
        up1 = tl_layer.ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = tl_layer.Conv2d(up1, 64, (3, 3), act=tf.nn.relu, name='uconv1_1')
        conv1 = tl_layer.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='uconv1_2')
        conv1 = tl_layer.Conv2d(conv1, n_out, (1, 1), act=tf.nn.sigmoid, name='uconv1')
    return conv1

# def u_net(x, is_train=False, reuse=False, pad='SAME', n_out=2):
#     """ Original U-Net for cell segmentataion
#     http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
#     Original x is [batch_size, 572, 572, ?], pad is VALID
#     """
#     from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2dLayer, ConcatLayer
#     nx = int(x._shape[1])
#     ny = int(x._shape[2])
#     nz = int(x._shape[3])
#     print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
#
#     w_init = tf.truncated_normal_initializer(stddev=0.01)
#     b_init = tf.constant_initializer(value=0.0)
#     with tf.variable_scope("u_net", reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#         inputs = InputLayer(x, name='inputs')
#
#         conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv1_1')
#         conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv1_2')
#         pool1 = MaxPool2d(conv1, (2, 2), padding=pad, name='pool1')
#
#         conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv2_1')
#         conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv2_2')
#         pool2 = MaxPool2d(conv2, (2, 2), padding=pad, name='pool2')
#
#         conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv3_1')
#         conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv3_2')
#         pool3 = MaxPool2d(conv3, (2, 2), padding=pad, name='pool3')
#
#         conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv4_1')
#         conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv4_2')
#         pool4 = MaxPool2d(conv4, (2, 2), padding=pad, name='pool4')
#
#         conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv5_1')
#         conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv5_2')
#
#         print(" * After conv: %s" % conv5.outputs)
#
#         up4 = DeConv2dLayer(conv5, 512, (3, 3), out_size = (nx/8, ny/8),
#                     strides=(2, 2), padding=pad, act=None,
#                     W_init=w_init, b_init=b_init, name='deconv4')
#         up4 = ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
#         conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv4_1')
#         conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv4_2')
#
#         up3 = DeConv2dLayer(conv4, 256, (3, 3), out_size = (nx/4, ny/4),
#                     strides=(2, 2), padding=pad, act=None,
#                     W_init=w_init, b_init=b_init, name='deconv3')
#         up3 = ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
#         conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv3_1')
#         conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv3_2')
#
#         up2 = DeConv2dLayer(conv3, 128, (3, 3), out_size=(nx/2, ny/2),
#                     strides=(2, 2), padding=pad, act=None,
#                     W_init=w_init, b_init=b_init, name='deconv2')
#         up2 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat2')
#         conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv2_1')
#         conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv2_2')
#
#         up1 = DeConv2dLayer(conv2, 64, (3, 3), out_size=(nx/1, ny/1),
#                     strides=(2, 2), padding=pad, act=None,
#                     W_init=w_init, b_init=b_init, name='deconv1')
#         up1 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat1')
#         conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv1_1')
#         conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv1_2')
#
#         conv1 = Conv2d(conv1, n_out, (1, 1), act=tf.nn.sigmoid, name='uconv1')
#         print(" * Output: %s" % conv1.outputs)
#
#         # logits0 = conv1.outputs[:,:,:,0]            # segmentataion
#         # logits1 = conv1.outputs[:,:,:,1]            # edge
#         # logits0 = tf.expand_dims(logits0, axis=3)
#         # logits1 = tf.expand_dims(logits1, axis=3)
#     return conv1


def u_net_bn_512(input_shape, is_train=False, batch_size=None, output_activation_func=tf.nn.sigmoid, pad='SAME', n_out=1):
    """image to image translation via conditional adversarial learning"""
    nx = int(input_shape[1])
    ny = int(input_shape[2])
    nz = int(input_shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))

    w_init = tf.initializers.TruncatedNormal(stddev=0.01)
    b_init = tf.initializers.Constant(value=0.0)
    gamma_init = tf.initializers.RandomNormal(1., 0.02)
    inputs = tl_layer.Input(input_shape, name='inputs')

    conv1 = tl_layer.Conv2d(64, (3, 3), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv1')(inputs)
    conv2 = tl_layer.Conv2d(128, (3, 3), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv2')(conv1)
    conv2 = tl_layer.BatchNorm(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn2')(conv2)

    conv3 = tl_layer.Conv2d(256, (3, 3), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv3')(conv2)
    conv3 = tl_layer.BatchNorm(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn3')(conv3)

    conv4 = tl_layer.Conv2d(512, (3, 3), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv4')(conv3)
    conv4 = tl_layer.BatchNorm(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn4')(conv4)

    conv5 = tl_layer.Conv2d(512, (3, 3), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv5')(conv4)
    conv5 = tl_layer.BatchNorm(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn5')(conv5)

    conv6 = tl_layer.Conv2d(512, (3, 3), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv6')(conv5)
    conv6 = tl_layer.BatchNorm(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn6')(conv6)

    conv7 = tl_layer.Conv2d(512, (3, 3), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv7')(conv6)
    conv7 = tl_layer.BatchNorm(act=lambda x: tf.nn.leaky_relu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn7')(conv7)

    conv8 = tl_layer.Conv2d(512, (3, 3), (2, 2), act=lambda x: tf.nn.leaky_relu(x, 0.2), padding=pad, W_init=w_init, b_init=b_init, name='conv8')(conv7)
    # print(" * After conv: %s" % conv8.shape)
    # exit()
    # print(nx/8)
    up7 = tl_layer.DeConv2dLayer(shape=(3, 3, 512, 512), outputs_shape=(batch_size, 4, 4, 512), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv7')(conv8)
    up7 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn7')(up7)

    # print(up6.outputs)
    up6 = tl_layer.Concat(concat_dim=3, name='concat6')([up7, conv7])
    up6 = tl_layer.DeConv2dLayer(shape=(3, 3, 512, 1024), outputs_shape=(batch_size, 8, 8, 512), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv6')(up6)
    up6 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn6')(up6)
    # print(up6.outputs)
    # exit()

    up5 = tl_layer.Concat(concat_dim=3, name='concat5')([up6, conv6])
    up5 = tl_layer.DeConv2dLayer(shape=(3, 3, 512, 1024), outputs_shape=(batch_size, 16, 16, 512), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv5')(up5)
    up5 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn5')(up5)
    # print(up5.outputs)
    # exit()

    up4 = tl_layer.Concat(concat_dim=3, name='concat4')([up5, conv5])
    up4 = tl_layer.DeConv2dLayer(shape=(3, 3, 512, 1024), outputs_shape=(batch_size, 32, 32, 512), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv4')(up4)
    up4 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn4')(up4)

    up3 = tl_layer.Concat(concat_dim=3, name='concat3')([up4, conv4])
    up3 = tl_layer.DeConv2dLayer(shape=(3, 3, 256, 1024), outputs_shape=(batch_size, 64, 64, 256), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv3')(up3)
    up3 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn3')(up3)

    up2 = tl_layer.Concat(concat_dim=3, name='concat2')([up3, conv3])
    up2 = tl_layer.DeConv2dLayer(shape=(3, 3, 128, 512), outputs_shape=(batch_size, 128, 128, 128), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv2')(up2)
    up2 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn2')(up2)

    up1 = tl_layer.Concat(concat_dim=3, name='concat1')([up2, conv2])
    up1 = tl_layer.DeConv2dLayer(shape=(3, 3, 64, 256), outputs_shape=(batch_size, 256, 256, 64), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv1')(up1)
    up1 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn1')(up1)

    up0 = tl_layer.Concat(concat_dim=3, name='concat0')([up1, conv1])
    up0 = tl_layer.DeConv2dLayer(shape=(3, 3, 64, 128), outputs_shape=(batch_size, 512, 512, 64), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv0')(up0)
    up0 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn0')(up0)
    # print(up0.outputs)
    # exit()

    out = tl_layer.Conv2d(n_out, (1, 1), act=output_activation_func, name='out')(up0)

    # print(" * Output: %s" % out.shape)
    # exit()
    network = tl.models.Model(inputs=inputs, outputs=out, name='u-net')

    return network


def u_net_bn_1024(input_shape, is_train=False, batch_size=None, output_activation_func=tf.nn.sigmoid, pad='SAME', n_out=1):
    """image to image translation via conditional adversarial learning"""
    nx = int(input_shape[1])
    ny = int(input_shape[2])
    nz = int(input_shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))

    w_init = tf.initializers.TruncatedNormal(stddev=0.01)
    b_init = tf.initializers.Constant(value=0.0)
    gamma_init = tf.initializers.RandomNormal(1., 0.02)
    # with tf.name_scope("u_net"):
    inputs = tl_layer.Input(input_shape, name='inputs')

    conv1 = tl_layer.Conv2d(32, (3, 3), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv1')(inputs)
    conv2 = tl_layer.Conv2d(64, (3, 3), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv2')(conv1)
    conv2 = tl_layer.BatchNorm(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn2')(conv2)

    conv3 = tl_layer.Conv2d(128, (3, 3), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv3')(conv2)
    conv3 = tl_layer.BatchNorm(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn3')(conv3)

    conv4 = tl_layer.Conv2d(256, (3, 3), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv4')(conv3)
    conv4 = tl_layer.BatchNorm(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn4')(conv4)

    conv5 = tl_layer.Conv2d(512, (3, 3), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv5')(conv4)
    conv5 = tl_layer.BatchNorm(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn5')(conv5)

    conv6 = tl_layer.Conv2d(512, (3, 3), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv6')(conv5)
    conv6 = tl_layer.BatchNorm(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn6')(conv6)

    conv7 = tl_layer.Conv2d(512, (3, 3), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv7')(conv6)
    conv7 = tl_layer.BatchNorm(act=lambda x: tf.nn.leaky_relu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn7')(conv7)

    conv8 = tl_layer.Conv2d(1024, (3, 3), (2, 2), act=lambda x: tf.nn.leaky_relu(x, 0.2), padding=pad, W_init=w_init, b_init=b_init, name='conv8')(conv7)
    # print(" * After conv: %s" % conv8.shape)
    # exit()Have a look at the alternative approach that's used in sqlalchemy: dependency injection:
    # print(nx/8)
    up7 = tl_layer.DeConv2dLayer(shape=(3, 3, 512, 1024), outputs_shape=(batch_size, 8, 8, 512), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv7')(conv8)
    up7 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn7')(up7)

    # print(up6.outputs)
    up6 = tl_layer.Concat(concat_dim=3, name='concat6')([up7, conv7])
    up6 = tl_layer.DeConv2dLayer(shape=(3, 3, 512, 1024), outputs_shape=(batch_size, 16, 16, 512), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv6')(up6)
    up6 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn6')(up6)
    # print(up6.outputs)
    # exit()

    up5 = tl_layer.Concat(concat_dim=3, name='concat5')([up6, conv6])
    up5 = tl_layer.DeConv2dLayer(shape=(3, 3, 512, 1024), outputs_shape=(batch_size, 32, 32, 512), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv5')(up5)
    up5 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn5')(up5)
    # print(up5.outputs)
    # exit()

    up4 = tl_layer.Concat(concat_dim=3, name='concat4')([up5, conv5])
    up4 = tl_layer.DeConv2dLayer(shape=(3, 3, 256, 1024), outputs_shape=(batch_size, 64, 64, 256), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv4')(up4)
    up4 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn4')(up4)

    up3 = tl_layer.Concat(concat_dim=3, name='concat3')([up4, conv4])
    up3 = tl_layer.DeConv2dLayer(shape=(3, 3, 128, 512), outputs_shape=(batch_size, 128, 128, 128), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv3')(up3)
    up3 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn3')(up3)

    up2 = tl_layer.Concat(concat_dim=3, name='concat2')([up3, conv3])
    up2 = tl_layer.DeConv2dLayer(shape=(3, 3, 64, 256), outputs_shape=(batch_size, 256, 256, 64), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv2')(up2)
    up2 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn2')(up2)

    up1 = tl_layer.Concat(concat_dim=3, name='concat1')([up2, conv2])
    up1 = tl_layer.DeConv2dLayer(shape=(3, 3, 32, 128), outputs_shape=(batch_size, 512, 512, 32), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv1')(up1)
    up1 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn1')(up1)

    up0 = tl_layer.Concat(concat_dim=3, name='concat0')([up1, conv1])
    up0 = tl_layer.DeConv2dLayer(shape=(3, 3, 32, 64), outputs_shape=(batch_size, 1024, 1024, 32), strides=(1, 2, 2, 1),
                        padding=pad, act=None, W_init=w_init, b_init=b_init, name='deconv0')(up0)
    up0 = tl_layer.BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn0')(up0)
    # print(up0.outputs)
    # exit()

    out = tl_layer.Conv2d(n_out, (1, 1), act=output_activation_func, name='out')(up0)

    # print(" * Output: %s" % out.shape)
    # exit()
    network = tl.models.Model(inputs=inputs, outputs=out, name='u-net')

    return network


def unet_keras(input_shape = (256,256,1)):

    inputs = Input(input_shape)
    conv1 = Conv2D(64, 3, padding= 'same', kernel_initializer= 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)

    conv2 = Conv2D(128, 3, padding= 'same', kernel_initializer= 'he_normal')(conv1)
    conv2 = Conv2D(128, 3, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNorm()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)

    conv3 = Conv2D(256, 3, padding= 'same', kernel_initializer= 'he_normal')(conv2)
    conv3 = Conv2D(256, 3, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNorm()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)

    conv4 = Conv2D(512, 3, padding= 'same', kernel_initializer= 'he_normal')(conv3)
    conv4 = Conv2D(512, 3, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNorm()(conv4)
    conv4 = LeakyReLU(alpha=0.2)(conv4)

    conv5 = Conv2D(1024, 3, padding= 'same', kernel_initializer= 'he_normal')(conv4)
    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNorm()(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)


def unet(input_size = (256,256,1)):
    inputs = tfk_layer.Input(input_size)
    conv1 = tfk_layer.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = tfk_layer.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = tfk_layer.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tfk_layer.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = tfk_layer.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = tfk_layer.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tfk_layer.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = tfk_layer.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = tfk_layer.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tfk_layer.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = tfk_layer.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = tfk_layer.Dropout(0.5)(conv4)
    pool4 = tfk_layer.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = tfk_layer.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = tfk_layer.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = tfk_layer.Dropout(0.5)(conv5)

    up6 = tfk_layer.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        tfk_layer.UpSampling2D(size = (2,2))(drop5))
    merge6 = tfk_layer.concatenate([drop4,up6], axis=3)
    conv6 = tfk_layer.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = tfk_layer.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = tfk_layer.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        tfk_layer.UpSampling2D(size = (2,2))(conv6))
    merge7 = tfk_layer.concatenate([conv3,up7], axis=3)
    conv7 = tfk_layer.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = tfk_layer.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = tfk_layer.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        tfk_layer.UpSampling2D(size = (2,2))(conv7))
    merge8 = tfk_layer.concatenate([conv2,up8], axis=3)
    conv8 = tfk_layer.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = tfk_layer.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = tfk_layer.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        tfk_layer.UpSampling2D(size = (2,2))(conv8))
    merge9 = tfk_layer.concatenate([conv1,up9], axis=3)
    conv9 = tfk_layer.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = tfk_layer.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = tfk_layer.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = tfk_layer.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = tfk_model.Model(inputs, conv10, name='u-net')

    return model


## old implementation
# def u_net_2d_64_1024_deconv(x, n_out=2):
#     from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2dLayer, ConcatLayer
#     nx = int(x._shape[1])
#     ny = int(x._shape[2])
#     nz = int(x._shape[3])
#     print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
#
#     w_init = tf.truncated_normal_initializer(stddev=0.01)
#     b_init = tf.constant_initializer(value=0.0)
#     inputs = InputLayer(x, name='inputs')
#
#     conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
#     conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
#     pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')
#
#     conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
#     conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
#     pool2 = MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')
#
#     conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
#     conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
#     pool3 = MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')
#
#     conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
#     conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')
#     pool4 = MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')
#
#     conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
#     conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')
#
#     print(" * After conv: %s" % conv5.outputs)
#
#     up4 = DeConv2dLayer(conv5, 512, (3, 3), out_size = (nx/8, ny/8), strides = (2, 2),
#                                 padding='same', act=None, W_init=w_init, b_init=b_init, name='deconv4')
#     up4 = ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
#     conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_1')
#     conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_2')
#
#     up3 = DeConv2dLayer(conv4, 256, (3, 3), out_size = (nx/4, ny/4), strides = (2, 2),
#                                 padding='same', act=None, W_init=w_init, b_init=b_init, name='deconv3')
#     up3 = ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
#     conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_1')
#     conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_2')
#
#     up2 = DeConv2dLayer(conv3, 128, (3, 3), out_size = (nx/2, ny/2), strides = (2, 2),
#                                 padding='same', act=None, W_init=w_init, b_init=b_init, name='deconv2')
#     up2 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat2')
#     conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_1')
#     conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_2')
#
#     up1 = DeConv2dLayer(conv2, 64, (3, 3), out_size = (nx/1, ny/1), strides = (2, 2),
#                                 padding='same', act=None, W_init=w_init, b_init=b_init, name='deconv1')
#     up1 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat1')
#     conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_1')
#     conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_2')
#
#     conv1 = Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')
#     print(" * Output: %s" % conv1.outputs)
#     outputs = tl.act.pixel_wise_softmax(conv1.outputs)
#     return conv1, outputs
#
#
# def u_net_2d_32_1024_upsam(x, n_out=2):
#     """
#     https://github.com/jocicmarko/ultrasound-nerve-segmentation
#     """
#     from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2dLayer, ConcatLayer
#     batch_size = int(x._shape[0])
#     nx = int(x._shape[1])
#     ny = int(x._shape[2])
#     nz = int(x._shape[3])
#     print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
#     ## define initializer
#     w_init = tf.truncated_normal_initializer(stddev=0.01)
#     b_init = tf.constant_initializer(value=0.0)
#     inputs = InputLayer(x, name='inputs')
#
#     conv1 = Conv2d(inputs, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
#     conv1 = Conv2d(conv1, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
#     pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')
#
#     conv2 = Conv2d(pool1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
#     conv2 = Conv2d(conv2, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
#     pool2 = MaxPool2d(conv2, (2,2), padding='SAME', name='pool2')
#
#     conv3 = Conv2d(pool2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
#     conv3 = Conv2d(conv3, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
#     pool3 = MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')
#
#     conv4 = Conv2d(pool3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
#     conv4 = Conv2d(conv4, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')
#     pool4 = MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')
#
#     conv5 = Conv2d(pool4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
#     conv5 = Conv2d(conv5, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')
#     pool5 = MaxPool2d(conv5, (2, 2), padding='SAME', name='pool6')
#
#     # hao add
#     conv6 = Conv2d(pool5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_1')
#     conv6 = Conv2d(conv6, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_2')
#
#     print(" * After conv: %s" % conv6.outputs)
#
#     # hao add
#     up7 = UpSampling2dLayer(conv6, (15, 15), is_scale=False, method=1, name='up7')
#     up7 =  ConcatLayer([up7, conv5], concat_dim=3, name='concat7')
#     conv7 = Conv2d(up7, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_1')
#     conv7 = Conv2d(conv7, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_2')
#
#     # print(nx/8,ny/8) # 30 30
#     up8 = UpSampling2dLayer(conv7, (2, 2), method=1, name='up8')
#     up8 = ConcatLayer([up8, conv4], concat_dim=3, name='concat8')
#     conv8 = Conv2d(up8, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_1')
#     conv8 = Conv2d(conv8, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_2')
#
#     up9 = UpSampling2dLayer(conv8, (2, 2), method=1, name='up9')
#     up9 = ConcatLayer([up9, conv3] ,concat_dim=3, name='concat9')
#     conv9 = Conv2d(up9, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_1')
#     conv9 = Conv2d(conv9, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_2')
#
#     up10 = UpSampling2dLayer(conv9, (2, 2), method=1, name='up10')
#     up10 = ConcatLayer([up10, conv2] ,concat_dim=3, name='concat10')
#     conv10 = Conv2d(up10, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv10_1')
#     conv10 = Conv2d(conv10, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv10_2')
#
#     up11 = UpSampling2dLayer(conv10, (2, 2), method=1, name='up11')
#     up11 = ConcatLayer([up11, conv1] ,concat_dim=3, name='concat11')
#     conv11 = Conv2d(up11, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv11_1')
#     conv11 = Conv2d(conv11, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv11_2')
#
#     conv12 = Conv2d(conv11, n_out, (1, 1), act=None, name='conv12')
#     print(" * Output: %s" % conv12.outputs)
#     outputs = tl.act.pixel_wise_softmax(conv12.outputs)
#     return conv10, outputs
#
#
# def u_net_2d_32_512_upsam(x, n_out=2):
#     """
#     https://github.com/jocicmarko/ultrasound-nerve-segmentation
#     """
#     from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2dLayer, ConcatLayer
#     batch_size = int(x._shape[0])
#     nx = int(x._shape[1])
#     ny = int(x._shape[2])
#     nz = int(x._shape[3])
#     print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
#     ## define initializer
#     w_init = tf.truncated_normal_initializer(stddev=0.01)
#     b_init = tf.constant_initializer(value=0.0)
#     inputs = InputLayer(x, name='inputs')
#     # inputs = Input((1, img_rows, img_cols))
#     conv1 = Conv2d(inputs, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
#     # print(conv1.outputs) # (10, 240, 240, 32)
#     # conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
#     conv1 = Conv2d(conv1, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
#     # print(conv1.outputs)    # (10, 240, 240, 32)
#     # conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
#     pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')
#     # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     # print(pool1.outputs)    # (10, 120, 120, 32)
#     # exit()
#     conv2 = Conv2d(pool1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
#     # conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
#     conv2 = Conv2d(conv2, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
#     # conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
#     pool2 = MaxPool2d(conv2, (2,2), padding='SAME', name='pool2')
#     # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#     conv3 = Conv2d(pool2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
#     # conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
#     conv3 = Conv2d(conv3, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
#     # conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
#     pool3 = MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')
#     # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     # print(pool3.outputs)   # (10, 30, 30, 64)
#
#     conv4 = Conv2d(pool3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
#     # print(conv4.outputs)    # (10, 30, 30, 256)
#     # conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
#     conv4 = Conv2d(conv4, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')
#     # print(conv4.outputs)    # (10, 30, 30, 256) != (10, 30, 30, 512)
#     # conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
#     pool4 = MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')
#     # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#     conv5 = Conv2d(pool4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
#     # conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
#     conv5 = Conv2d(conv5, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')
#     # conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
#     # print(conv5.outputs)    # (10, 15, 15, 512)
#     print(" * After conv: %s" % conv5.outputs)
#     # print(nx/8,ny/8) # 30 30
#     up6 = UpSampling2dLayer(conv5, (2, 2), name='up6')
#     # print(up6.outputs)  # (10, 30, 30, 512) == (10, 30, 30, 512)
#     up6 = ConcatLayer([up6, conv4], concat_dim=3, name='concat6')
#     # print(up6.outputs)  # (10, 30, 30, 768)
#     # up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
#     conv6 = Conv2d(up6, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_1')
#     # conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
#     conv6 = Conv2d(conv6, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_2')
#     # conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)
#
#     up7 = UpSampling2dLayer(conv6, (2, 2), name='up7')
#     up7 = ConcatLayer([up7, conv3] ,concat_dim=3, name='concat7')
#     # up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
#     conv7 = Conv2d(up7, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_1')
#     # conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
#     conv7 = Conv2d(conv7, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_2')
#     # conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)
#
#     up8 = UpSampling2dLayer(conv7, (2, 2), name='up8')
#     up8 = ConcatLayer([up8, conv2] ,concat_dim=3, name='concat8')
#     # up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
#     conv8 = Conv2d(up8, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_1')
#     # conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
#     conv8 = Conv2d(conv8, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_2')
#     # conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
#
#     up9 = UpSampling2dLayer(conv8, (2, 2), name='up9')
#     up9 = ConcatLayer([up9, conv1] ,concat_dim=3, name='concat9')
#     # up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
#     conv9 = Conv2d(up9, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_1')
#     # conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
#     conv9 = Conv2d(conv9, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_2')
#     # conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
#
#     conv10 = Conv2d(conv9, n_out, (1, 1), act=None, name='conv9')
#     # conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
#     print(" * Output: %s" % conv10.outputs)
#     outputs = tl.act.pixel_wise_softmax(conv10.outputs)
#     return conv10, outputs


if __name__ == "__main__":
    pass
    # main()



















#
