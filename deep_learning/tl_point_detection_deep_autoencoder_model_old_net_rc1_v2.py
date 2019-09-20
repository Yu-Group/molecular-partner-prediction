#####################################################################
# 
# Author: Xiongtao Ruan
# 
# Copyright (C) 2012-2018 Murphy Lab
# Computational Biology Department
# School of Computer Science
# Carnegie Mellon University
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301, USA.
# 
# For additional information visit http://murphylab.web.cmu.edu or
# send email to murphy@cmu.edu
# 
#####################################################################


import tensorflow as tf
import tensorflow_addons as tfa
import tensorlayer as tl
from layers_v2 import *

# tf.compat.v1.disable_eager_execution()


''' 03/29/2018 copied from tl_HPA_deep_autoencoder_model_old_net_rc0.py '''
''' 04/16/2018 change is_train=True for the first block as is_train '''
''' 05/01/2018 modify decoder network'''

def deep_conv_autoencoder_model(input_shape, filter_n1 = 1, filter_n2 = 2, filter_n3 = 4, filter_n4 = 8, filter_n5 = 16,
                                filter_n6 = 32, filter_n7 =32, filter_n8 =32, filter_size= 3,
                                latent_dim = 7, network_type = 'only_cell', layer_type = 'FC',
                                lambda_param = 0, lambda_2_param = 0, reuse=False, is_train=True):
    
    W_init = tf.initializers.GlorotUniform()

    l_img_in = tl.layers.Input(input_shape, name='input_layer')

    l_img_mg1 = residual_net_encode_module(l_img_in, filter_n1=1, filter_n2=filter_n1, filter_size=filter_size,
                                           is_train=is_train, name='img_resnet_block_1')
    l_img_mg2 = residual_net_encode_module(l_img_mg1, filter_n1=filter_n1, filter_n2=filter_n2, filter_size=filter_size,
                                           is_train=is_train, name='img_resnet_block_2')
    l_img_mg3 = residual_net_encode_module(l_img_mg2, filter_n1=filter_n2, filter_n2=filter_n3, filter_size=filter_size,
                                           is_train=is_train, name='img_resnet_block_3')
    l_img_mg4 = residual_net_encode_module(l_img_mg3, filter_n1=filter_n3, filter_n2=filter_n4, filter_size=filter_size,
                                           is_train=is_train, name='img_resnet_block_4')
    l_img_mg5 = residual_net_encode_module(l_img_mg4, filter_n1=filter_n4, filter_n2=filter_n5, filter_size=filter_size,
                                           is_train=is_train, name='img_resnet_block_5')
    l_img_mg6 = residual_net_encode_module(l_img_mg5, filter_n1=filter_n5, filter_n2=filter_n6, filter_size=filter_size,
                                           is_train=is_train, name='img_resnet_block_6')
    l_img_mg7 = residual_net_encode_module(l_img_mg6, filter_n1=filter_n6, filter_n2=filter_n7, filter_size=filter_size,
                                           is_train=is_train, name='img_resnet_block_7')
    l_img_mg8 = residual_net_encode_module(l_img_mg7, filter_n1=filter_n7, filter_n2=filter_n8, filter_size=filter_size,
                                           is_train=is_train, name='img_resnet_block_8')

    l_reshape = tl.layers.Flatten(name='flatten')(l_img_mg8)
    # l_dropout = tl.layers.DropoutLayer(l_reshape, keep=0.9, is_fix=True, is_train=is_train, name='drop1')
    if layer_type == 'FC':
        encoded = tl.layers.Dense(n_units=latent_dim, W_init=W_init, name='encoded')(l_reshape)

    # l_dropout_1 = tl.layers.DropoutLayer(encoded, keep=0.9, is_fix=True, is_train=is_train, name='drop2')
    l_dense3 = tl.layers.Dense(n_units=l_reshape.get_shape().as_list()[1], W_init=W_init, name='dense3')(encoded)
    # print(tl.layers.get_output_shape(l_pool5))
    l_reshape_1 = tl.layers.Reshape(shape=[-1] + l_img_mg8.get_shape().as_list()[1:], name='reshape1')(l_dense3)
    # l_dropout_2 = tl.layers.DropoutLayer(l_reshape_1, keep=0.9, is_fix=True, is_train=is_train, name='drop3')

    l_dc_out1 = residual_net_decode_module(l_reshape_1, filter_n1=filter_n8, filter_n2=filter_n8, filter_size=filter_size, is_train=is_train, name='dc_resnet_block_1')
    l_dc_out2 = residual_net_decode_module(l_dc_out1, filter_n1=filter_n8, filter_n2=filter_n7, filter_size=filter_size, is_train=is_train, name='dc_resnet_block_2')
    l_dc_out3 = residual_net_decode_module(l_dc_out2, filter_n1=filter_n7, filter_n2=filter_n6, filter_size=filter_size, is_train=is_train, name='dc_resnet_block_3')
    l_dc_out4 = residual_net_decode_module(l_dc_out3, filter_n1=filter_n6, filter_n2=filter_n5, filter_size=filter_size, is_train=is_train, name='dc_resnet_block_4')
    l_dc_out5 = residual_net_decode_module(l_dc_out4, filter_n1=filter_n5, filter_n2=filter_n4, filter_size=filter_size, is_train=is_train, name='dc_resnet_block_5')
    l_dc_out6 = residual_net_decode_module(l_dc_out5, filter_n1=filter_n4, filter_n2=filter_n3, filter_size=filter_size, is_train=is_train, name='dc_resnet_block_6')
    l_dc_out7 = residual_net_decode_module(l_dc_out6, filter_n1=filter_n3, filter_n2=filter_n2, filter_size=filter_size, is_train=is_train, name='dc_resnet_block_7')
    l_dc_out8 = residual_net_decode_module(l_dc_out7, filter_n1=filter_n2, filter_n2=filter_n1, filter_size=filter_size, is_train=is_train, name='dc_resnet_block_8')

    if network_type == 'only_cell':
        decoded = tl.layers.Conv2dLayer(shape=(filter_size, filter_size, filter_n1, 1), strides=(1, 1, 1, 1), W_init=W_init, padding='SAME', name='decoded')(l_dc_out8)
    elif network_type == 'cell_nuc':
        decoded = tl.layers.Conv2dLayer(shape=(filter_size, filter_size, filter_n1, 3), strides=(1, 1, 1, 1), W_init=W_init, padding='SAME', name='decoded')(l_dc_out8)

    reshape_shape = [-1] + l_img_mg8.get_shape().as_list()[1:]
    encoder = tl.models.Model(inputs=l_img_in, outputs=encoded, name='encoder')
    decoder = tl.models.Model(inputs=encoded, outputs=decoded, name='decoder')
    autoencoder = tl.models.Model(inputs=l_img_in, outputs=decoded, name='ae')
    return autoencoder, encoder, decoder, reshape_shape


def deep_conv_autoencoder_decode_model(input_var, filter_n1 = 1, filter_n2 = 2, filter_n3 = 4, filter_n4 = 8, filter_n5 = 16, filter_n6 = 32, filter_n7 =32, filter_size= 3,
                          latent_dim = 7, network_type = 'only_cell', layer_type = 'FC', reshape_shape = [-1, 4, 4, 16], reuse=False, is_train=True):

    num_unit = np.prod(reshape_shape[1:])
    W_init = tf.initializers.GlorotUniform()
    tl.layers.set_name_reuse(reuse)

    l_img_in = tl.layers.InputLayer(input_var, name='encode_input_layer')
    l_dense3 = tl.layers.DenseLayer(l_img_in, n_units=num_unit, W_init=W_init, name='dense3')
    # print(tl.layers.get_output_shape(l_pool5))
    l_reshape_1 = tl.layers.ReshapeLayer(l_dense3, shape=reshape_shape, name='reshape1')
    # l_dropout_2 = tl.layers.DropoutLayer(l_reshape_1, keep=0.9, is_fix=True, is_train=is_train, name='drop3')

    l_dc_out1 = residual_net_decode_module(l_reshape_1, filter_n1=filter_n7, filter_n2=filter_n7, filter_size=filter_size, is_train=is_train, name='dc_resnet_block_1')
    l_dc_out2 = residual_net_decode_module(l_dc_out1, filter_n1=filter_n7, filter_n2=filter_n6, filter_size=filter_size, is_train=is_train, name='dc_resnet_block_2')
    l_dc_out3 = residual_net_decode_module(l_dc_out2, filter_n1=filter_n6, filter_n2=filter_n5, filter_size=filter_size, is_train=is_train, name='dc_resnet_block_3')
    l_dc_out4 = residual_net_decode_module(l_dc_out3, filter_n1=filter_n5, filter_n2=filter_n4, filter_size=filter_size, is_train=is_train, name='dc_resnet_block_4')
    l_dc_out5 = residual_net_decode_module(l_dc_out4, filter_n1=filter_n4, filter_n2=filter_n3, filter_size=filter_size, is_train=is_train, name='dc_resnet_block_5')
    l_dc_out6 = residual_net_decode_module(l_dc_out5, filter_n1=filter_n3, filter_n2=filter_n2, filter_size=filter_size, is_train=is_train, name='dc_resnet_block_6')
    l_dc_out7 = residual_net_decode_module(l_dc_out6, filter_n1=filter_n2, filter_n2=filter_n1, filter_size=filter_size, is_train=is_train, name='dc_resnet_block_7')

    if network_type == 'only_cell':
        decoded = tl.layers.Conv2dLayer(l_dc_out7, shape = [filter_size, filter_size, filter_n1, 1], strides=(1, 1, 1, 1), W_init=W_init, padding='SAME', name='decoded')
    elif network_type == 'cell_nuc':
        decoded = tl.layers.Conv2dLayer(l_dc_out7, shape = [filter_size, filter_size, filter_n1, 3], strides=(1, 1, 1, 1), W_init=W_init, padding='SAME', name='decoded')

    return  decoded

