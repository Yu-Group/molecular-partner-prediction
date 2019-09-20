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
import numpy as np
import scipy.spatial
# from binary_image_hausdorff_distance import *


def costum_laplace_loss(a, b):
    G_ker = tf.constant(np.transpose(np.array([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]]).astype(np.float32), [2, 3, 0, 1]))
    # a = tf.reshape(a, [-1] + a.shape.as_list()[1:])
    # b = tf.reshape(b, [-1] + b.shape.as_list()[1:])

    ga = tf.nn.conv2d(a, G_ker, strides=[1, 1, 1, 1], padding = 'SAME')

    gb = tf.nn.conv2d(b, G_ker, strides=[1, 1, 1, 1], padding = 'SAME')

    return tf.reduce_mean(tf.nn.relu(1- a * (2 * b - 1.0))) + tf.reduce_mean(tf.abs((ga-gb) * gb)) * 0.1

def multi_resolution_loss_function(predictions, labels, downsample_times):
    loss = 0
    for i in range(downsample_times):
        cur_predict = tf.nn.max_pool(predictions, ksize = [1, 2 ** i, 2 **i, 1], strides=[1, 2 ** i, 2 ** i, 1], padding='VALID')
        cur_labels = tf.nn.max_pool(labels, ksize = [1, 2 ** i, 2 **i, 1], strides=[1, 2 ** i, 2 ** i, 1], padding='VALID')
        loss = loss + costum_laplace_loss(cur_predict, cur_labels) *  (2 ** (-i))

    return loss

def hinge_loss_function(predictions, labels):
    loss = 0
    # predictions = T.log(predictions / (1 - predictions))
    loss = tf.reduce_mean(tf.nn.relu(1.0 - predictions * (2 * labels - 1.0)))
    return loss

def general_hinge_loss_function(predictions, labels, factor=100):
    loss = 0
    # predictions = T.log(predictions / (1 - predictions))
    base_loss = tf.nn.relu(1.0 - predictions * (2 * labels - 1.0))
    loss = tf.reduce_mean(base_loss * labels * factor + base_loss * (1 - labels))
    return loss

def reverse_hinge_loss_function(predictions, labels):
    loss = 0
    # predictions = T.log(predictions / (1 - predictions))
    base_loss = tf.nn.relu(1.0 - predictions * (2 * labels - 1.0))
    loss = tf.reduce_mean(base_loss * labels * 1 + base_loss * (1 - labels) * 100)
    return loss

def general_binary_cross_entropy(predictions, labels, factor=1.):
    loss = 0
    # predictions = T.log(predictions / (1 - predictions))
    epsilon = 1e-8
    loss = -tf.reduce_mean(tf.log(predictions + epsilon) * labels * factor + tf.log(1. - predictions + epsilon) * (1. - labels))

    return loss

def hinge_loss_3d_function(predictions, labels):
    loss = 0
    # predictions = T.log(predictions / (1 - predictions))
    loss = tf.reduce_mean(tf.nn.relu(1.0 - predictions * (2 * labels - 1.0)))
    return loss

def softmax_cross_entropy(predict, target, axis=3):
    loss = 0
    A0, A1, A2 = tf.split(predict, [1, 1, 1], axis)
    B0 = tf.to_float(tf.equal(target, 0))
    B1 = tf.to_float(tf.equal(target, 1))
    B2 = tf.to_float(tf.equal(target, 2))
    epsilon = 1e-8
    loss = -tf.reduce_mean(tf.log(A0 + epsilon) * B0 + tf.log(A1 + epsilon) * B1 + tf.log(A2 + epsilon) * B2)

    return loss

def multiclass_hinge_loss_function(predictions, labels):
    '''Based on tensorflow function tf.contrib.kernel_methods.sparse_multiclass_hinge_loss'''
    predictions_reshape = tf.reshape(predictions, [-1, predictions.shape.as_list()[-1]])
    labels_reshape = tf.reshape(labels, [-1, labels.shape.as_list()[-1]])
    loss = tf.reduce_mean(tf.contrib.kernel_methods.sparse_multiclass_hinge_loss(labels=labels_reshape, logits=predictions_reshape))

    return loss

def multiclasss_multi_resolution_loss_function(predictions, labels, downsample_times = 4):
    loss = 0
    for i in range(downsample_times):
        cur_predict = tf.nn.max_pool(predictions, ksize = [1, 2 ** i, 2 **i, 1], strides=[1, 2 ** i, 2 ** i, 1], padding='VALID')
        cur_labels = tf.nn.max_pool(labels, ksize = [1, 2 ** i, 2 **i, 1], strides=[1, 2 ** i, 2 ** i, 1], padding='VALID')
        loss = loss + softmax_cross_entropy(cur_predict, cur_labels) *  (2 ** (-i))

    return loss

def F1_score_accuaracy(predict, target):
    A1 = tf.to_float(tf.equal(predict, 1))
    B1 = tf.to_float(tf.equal(target, 1))
    F1_1 = 2 * tf.reduce_sum(A1 * B1) / (tf.reduce_sum(A1) + tf.reduce_sum(B1))
    A2 = tf.to_float(tf.equal(predict, 2))
    B2 = tf.to_float(tf.equal(target, 2))
    F1_2 = 2 * tf.reduce_sum(A2 * B2) / (tf.reduce_sum(A2) + tf.reduce_sum(B2))
    return (F1_1 + F1_2) / 2

def weighted_average_accuaracy(predict, target, method = 1, weights=[1, 0.5, 0]):
    if method == 1:
        Area = tf.to_float(target >= 1)
    elif method == 2:
        Area = tf.to_float((predict + target) >= 1)

    A0 = tf.to_float(tf.equal(predict, 0)) * Area
    A1 = tf.to_float(tf.equal(predict, 1))
    A2 = tf.to_float(tf.equal(predict, 2))

    B0 = tf.to_float(tf.equal(target, 0)) * Area
    B1 = tf.to_float(tf.equal(target, 1))
    B2 = tf.to_float(tf.equal(target, 2))

    w0 = weights[0]
    w1 = weights[1]
    w2 = weights[2]
    accuracy = tf.reduce_sum(A1 * B1) * w0 + tf.reduce_sum(A2 * B2) * w0 \
               + tf.reduce_sum(A1 * B2) * w1 + tf.reduce_sum(A2 * B1) * w1 \
               + tf.reduce_sum(A1 * B0) * w2 + tf.reduce_sum(A2 * B0) * w2 \
               + tf.reduce_sum(A0 * B1) * w2 + tf.reduce_sum(A0 * B2) * w2

    accuracy = accuracy / tf.reduce_sum(Area)
    return accuracy

def nuc_cyto_separate_accuaracies(predict, target):
    rank = len(target.get_shape())
    cyto_predict = tf.to_float(predict >= 1)
    nuc_predict = tf.to_float(predict >= 2)
    cyto_target = tf.to_float(target >= 1)
    nuc_target = tf.to_float(target >= 2)

    cyto_accuracy = tf.reduce_mean(tf.to_float(tf.equal(cyto_predict, cyto_target)))
    nuc_accuracy = tf.reduce_mean(tf.to_float(tf.equal(nuc_predict, nuc_target)))

    cyto_reconst_1 = tf.reduce_mean(tf.reduce_sum(tf.abs(cyto_predict - cyto_target), axis = range(1, rank)) / tf.reduce_sum(cyto_target, axis = range(1, rank)))
    nuc_reconst_1 = tf.reduce_mean(tf.reduce_sum(tf.abs(nuc_predict - nuc_target), axis = range(1, rank)) / tf.reduce_sum(nuc_target, axis = range(1, rank)))
    cyto_reconst_2 = tf.reduce_mean(tf.reduce_sum(tf.abs(cyto_predict - cyto_target), axis = range(1, rank)) / tf.reduce_sum(tf.to_float(cyto_target + cyto_predict >=0.5), axis = range(1, rank)))
    nuc_reconst_2 = tf.reduce_mean(tf.reduce_sum(tf.abs(nuc_predict - nuc_target), axis = range(1, rank)) / tf.reduce_sum(tf.to_float(nuc_target + nuc_predict >=0.5), axis = range(1, rank)))

    return cyto_accuracy, nuc_accuracy, cyto_reconst_1, nuc_reconst_1, cyto_reconst_2, nuc_reconst_2


def outline_hausdorff_distance(predict, target):
    ''' Calculate the hausdorff distance for each outline between target and prediciton '''
    ''' 10/18/2018 use cKDTree based method for the computing'''
    batch_size = target.shape[0]
    hd_array = np.zeros(batch_size)
    for ind in range(len(target)):
        # hd1 = scipy.spatial.distance.directed_hausdorff(predict[ind], target[ind])[0]
        # hd2 = scipy.spatial.distance.directed_hausdorff(target[ind], predict[ind])[0]
        tree = scipy.spatial.cKDTree(predict[ind])
        hd1 = np.max(tree.query(target[ind])[0])
        tree = scipy.spatial.cKDTree(target[ind])
        hd2 = np.max(tree.query(predict[ind])[0])
        hd_array[ind] = max(hd1, hd2)
    return hd_array.astype(np.float32)


def iterate_minibatches_single_data(inputs, batchsize, shuffle=False, throw_last_incomplete_batch=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for ind in range(0, len(inputs) // batchsize + 1 * (len(inputs) % batchsize != 0)):
        if throw_last_incomplete_batch:
            if (ind+1) *batchsize > len(inputs):
                break
        if shuffle:
            excerpt = indices[ind * batchsize: min((ind+1) *batchsize, len(inputs))]
        else:
            excerpt = slice(ind * batchsize, min((ind+1) *batchsize, len(inputs)))
        yield inputs[excerpt], excerpt


def iterate_minibatches(inputs, targets, batchsize, shuffle=False, throw_last_incomplete_batch=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for ind in range(0, len(inputs) // batchsize + 1 * (len(inputs) % batchsize != 0)):
        if throw_last_incomplete_batch:
            if (ind+1) *batchsize > len(inputs):
                break
        if shuffle:
            excerpt = indices[ind * batchsize: min((ind+1) *batchsize, len(inputs))]
        else:
            excerpt = slice(ind * batchsize, min((ind+1) *batchsize, len(inputs)))
        yield inputs[excerpt], targets[excerpt], excerpt


def iterate_minibatches_two_inputs(inputs_1, inputs_2, targets_1, targets_2, batchsize, shuffle=False):
    assert len(inputs_1) == len(targets_1)
    if shuffle:
        indices = np.arange(len(inputs_1))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs_1) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs_1[excerpt], inputs_2[excerpt], targets_1[excerpt], targets_2[excerpt], excerpt



def shape_evolve_2d_image_energy(source, target, image_evolve):

    num_img = image_evolve.shape[0]
    
    energy_mat = np.zeros((num_img, 2))

    for i in range(num_img):
        cur_image = image_evolve[0]
        hd_s, _, _ = hausdorff_distance_for_bindary_images(source, cur_image)
        hd_t, _, _ = hausdorff_distance_for_bindary_images(target, cur_image)
        energy_mat[i,:] = [hd_s, hd_t]

    energy = np.mean(energy_mat)
    hd, _, _ = hausdorff_distance_for_bindary_images(source, target)
    return energy, energy_mat, hd
   


def count_tensorlayer_network_parameter_number(network, mode='tensorlayer'):
    # Helper function to count the number of parameter in the network

    if mode == 'tensorlayer':
        all_weights = network.all_weights
    elif mode == 'tensorflow':
        all_weights = network.weights

    trainable_weights = network.trainable_weights

    all_weights_number = 0
    for weight in all_weights:
        all_weights_number += tf.reduce_prod(weight.shape)

    trainable_weights_number = 0
    for weight in trainable_weights:
        trainable_weights_number += tf.reduce_prod(weight.shape)

    return all_weights_number, trainable_weights_number

    
