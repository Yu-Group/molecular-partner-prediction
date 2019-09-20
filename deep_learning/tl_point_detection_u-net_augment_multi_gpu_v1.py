import multiprocessing

import numpy as np

import os, sys, time, errno
import matplotlib

import matplotlib.pylab as plt
import scipy
import scipy.io as sio
from scipy.ndimage.filters import maximum_filter
import h5py
import hdf5storage
from mpi4py import MPI

import tensorflow as tf
import tensorlayer as tl

import horovod.tensorflow as hvd

from chunk_lock_clean import Chunk_lock_clean
from process_options_structure import *
from utils import *
from tl_point_detection_deep_autoencoder_model_old_net_rc1_v2 import *
from tf_image_mask_augment import augment
# from tensorlayer_image_mask_augment import augment
from u_net import *

np.random.seed(1254)
matplotlib.use('Agg')
tf.compat.v1.random.set_random_seed(1254)
sys.path.append("./")
sys.path.append("./deep_learning")


'''08/23/2019 change to use tensorlayer-based image augmentation and tensorflow data storage'''


def deep_conv_autoencoder(filter_n1=1, filter_n2=2, filter_n3=4, filter_n4=8, filter_n5=16, filter_n6=32, filter_n7=64,
                          filter_n8=128, filter_size=3, filter_size_1=3, filter_size_2=1, intermediate_dim=128,
                          latent_dim=7, batch_size=4, nb_epoch=1000, base_lr=1e-3, print_freq=10, train_num=10000,
                          lr_method='step', print_epoch=900, data_filename='', data_type='original', layer_type='FC',
                          data_option='original', loss_func='binary_cross_entropy', cell_component='cell',
                          do_augmentation=True, result_directory='', model_suffix='', image_size=1024):

    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    image_dir = "%simage_dir/" % result_directory
    model_dir = "%smodel_dir/" % result_directory
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # input using images

    # input_var = tf.compat.v1.placeholder(tf.float32, shape=[None, 1024, 1024, 1], name='input')
    # target_var = tf.compat.v1.placeholder(tf.float32, shape=[None, 1024, 1024, 1], name='target')
    # if do_augmentation:
    #     aug_input_var, aug_target_var = augment(input_var, target_var, horizontal_flip=True, vertical_flip=True, rotate=45, crop_probability=0.2, mixup=0)
    # else:
    #     aug_input_var = input_var
    #     aug_target_var = target_var

    # input_shape = [batch_size, 512, 512, 1]
    # network = u_net_bn(input_shape, batch_size=batch_size, pad='SAME', n_out=1)
    if loss_func == 'hinge':
        output_activation_func = None
        loss_func_handle = lambda yt, yp: tf.reduce_mean(tf.reduce_sum(tf.losses.hinge(yt, yp), axis=(1, )))
        intensity_thrsh = 0
    elif loss_func == 'binary_cross_entropy':
        output_activation_func = tf.nn.sigmoid
        loss_func_handle = tl.cost.binary_cross_entropy
        intensity_thrsh = 0.5
    elif loss_func == 'dice_coe':
        output_activation_func = tf.nn.sigmoid
        loss_func_handle = lambda yt, yp: 1 - tl.cost.dice_coe(yp, yt, axis=(0, 1, 2, 3))
        intensity_thrsh = 0.5

    if image_size == 1024:
        input_shape = [batch_size, 1024, 1024, 1]
        network = u_net_bn_1024(input_shape, batch_size=batch_size, output_activation_func=output_activation_func, pad='SAME', n_out=1)
    elif image_size == 512:
        input_shape = [batch_size, 512, 512, 1]
        network = u_net_bn_512(input_shape, batch_size=batch_size, output_activation_func=output_activation_func, pad='SAME', n_out=1)

    train_weights = network.trainable_weights
    all_weights_number, trainable_weights_number = count_tensorlayer_network_parameter_number(network)
    # autoencoder.print_layers()
    if hvd.rank() == 0:
        print(network)
        print('Number of parameters: {} , number of trainable parameters: {}'.format(all_weights_number,
                                                                                     trainable_weights_number))

    def performance_criterion_function(target_var, prediction, intensity_thrsh=0.0,
                                       loss_func_handle=tf.losses.binary_crossentropy):

        # cost = tf.compat.v1.losses.hinge_loss(target_var, prediction)
        cost = loss_func_handle(target_var, prediction)
        predict_image = tf.cast(prediction >= intensity_thrsh, tf.float32)
        prediction_accuracy = tf.reduce_mean(tf.cast(tf.equal(predict_image, target_var), tf.float32))
        prediction_accuracy_1 = tf.reduce_mean(tf.reduce_sum(tf.abs(predict_image - target_var), axis=range(1, 4))
                                               / tf.reduce_sum(target_var, axis=range(1, 4)))
        prediction_accuracy_2 = tf.reduce_mean(tf.reduce_sum(tf.abs(predict_image - target_var), axis=range(1, 4))
                                               / tf.reduce_sum(tf.cast(predict_image + target_var >= 0.5, tf.float32),
                                                               axis=range(1, 4)))
        tp_image_sum = tf.reduce_sum(predict_image * target_var, axis=range(1, 4))
        precision = tf.reduce_mean(tp_image_sum / tf.reduce_sum(predict_image, axis=range(1, 4)))
        recall = tf.reduce_mean(tp_image_sum / tf.reduce_sum(target_var, axis=range(1, 4)))
        f1_score = tf.reduce_mean(2 * tp_image_sum / tf.reduce_sum(predict_image + target_var, axis=range(1, 4)))

        return cost, prediction_accuracy, prediction_accuracy_1, prediction_accuracy_2, precision, recall, f1_score

    data = h5py.File(data_filename, 'r')
    image_data = np.array(data['raw_image_mat'])
    image_data = image_data.reshape(image_data.shape[0], image_data.shape[1], image_data.shape[2], 1)
    object_data = np.array(data['object_mat'])
    object_data = object_data.reshape(object_data.shape[0], object_data.shape[1], object_data.shape[2], 1)

    print(image_data.shape)
    del data
    # change so that the new data set can be used for the running
    if data_option == 'size_1024':
        test_ind = 6000
    elif data_option == 'size_512':
        test_ind = 24000
    elif data_option == 'size_512_small':
        test_ind = 6000

    if 'cell_lines' in data_type:
        X_train = image_data[:train_num, ]
        X_test = image_data[10000:, ]
    else:
        X_train = image_data[:train_num, ]
        Y_train = object_data[:train_num, ]
        X_test = image_data[test_ind:, ]
        Y_test = object_data[test_ind:, ]
    del image_data, object_data

    # downsample images
    # sum_func = lambda x: x[:, ::2, ::2] + x[:, ::2, 1::2] + x[:, 1::2, ::2] + x[:, 1::2, 1::2]
    # X_train = sum_func(X_train) / 4
    # Y_train = sum_func(Y_train) > 0.0
    # X_test = sum_func(X_test) / 4
    # Y_test = sum_func(Y_test) > 0.0

    # saver = tf.compat.v1.train.Saver()
    opt = tf.optimizers.Adam(clipvalue=1.0)
    # opt.clipvalue_1 = 1
    # opt = hvd.DistributedOptimizer(opt)

    def generator_train():
        inputs = X_train
        targets = Y_train
        if len(inputs) != len(targets):
            raise AssertionError("The length of inputs and targets should be equal")
        for _input, _target in zip(inputs, targets):
            # yield _input.encode('utf-8'), _target.encode('utf-8')
            yield _input, _target

    def generator_test():
        inputs = X_test
        targets = Y_test
        if len(inputs) != len(targets):
            raise AssertionError("The length of inputs and targets should be equal")
        for _input, _target in zip(inputs, targets):
            # yield _input.encode('utf-8'), _target.encode('utf-8')
            yield _input, _target

    # dataset API and augmentation
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.float32))
    shuffle_buffer_size = 1000
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.shard(hvd.size(), hvd.rank())
    train_ds = train_ds.prefetch(buffer_size=500)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.map(lambda x, y: augment(x, y, horizontal_flip=True, vertical_flip=True, rotate=45,
                           crop_probability=0.2), num_parallel_calls=multiprocessing.cpu_count())

    # dataset for inference
    train_inf_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.float32))
    train_inf_ds = train_inf_ds.shard(hvd.size(), hvd.rank())
    train_inf_ds = train_inf_ds.prefetch(buffer_size=500)
    train_inf_ds = train_inf_ds.batch(batch_size, drop_remainder=True)

    test_ds = tf.data.Dataset.from_generator(generator_test, output_types=(tf.float32, tf.float32))
    test_ds = test_ds.shard(hvd.size(), hvd.rank())
    # test_ds = test_ds.shuffle(shuffle_buffer_size)
    # test_ds = test_ds.map(_map_fn_test, num_parallel_calls=multiprocessing.cpu_count())
    # test_ds = test_ds.repeat(n_epoch)
    test_ds = test_ds.prefetch(buffer_size=1500)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)

    print("starting training...")
    num_best_model_to_keep = 3
    best_train_error_mat = np.ones((num_best_model_to_keep, 1)) * 1e10
    best_model_inds = [0] * num_best_model_to_keep
    log_infos = np.zeros((nb_epoch, 5))
    first_batch = True
    inference_batch_size = 50
    for epoch in range(nb_epoch):
        if lr_method == 'old':
            if epoch > 1 and epoch % 20 == 0:
                lr = lr > min_lr and base_lr * (lr_decay ** epoch) or min_lr
        elif lr_method == 'step':
            if epoch < 10:
                lr = 1e-7 + epoch / 10 * (5e-4 - 1e-7)
            elif epoch < nb_epoch * 0.75:
                lr = 5e-4
                # lr = 5e-4
            elif epoch < nb_epoch * 0.9:
                lr = 5e-5
            elif epoch < nb_epoch:
                lr = 5e-6

        if epoch < nb_epoch * 0.9:
            print_freq = 20
        elif epoch < nb_epoch:
            print_freq = 2

        opt.learning_rate = lr * hvd.size()
        start_time = time.time()

        aug_train_loss = 0
        batched_nums = 0
        for X_train_a, Y_train_a in train_ds:
            with tf.GradientTape() as tape:
                _logits = network(X_train_a, is_train=True)
                # _loss = tf.compat.v1.losses.hinge_loss(Y_train_a, _logits)
                # _loss = 1 - tl.cost.dice_coe(_logits, Y_train_a, axis=(0, 1, 2, 3))
                # loss = tl.cost.binary_cross_entropy(_logits, Y_train_a)
                _loss = loss_func_handle(Y_train_a, _logits)

            tape = hvd.DistributedGradientTape(tape)
            grad = tape.gradient(_loss, train_weights)
            opt.apply_gradients(zip(grad, train_weights))
            # opt.get_gradients(_loss, network.trainable_variables)

            if first_batch:
                hvd.broadcast_variables(network.all_weights, root_rank=0)
                hvd.broadcast_variables(opt.variables(), root_rank=0)
                first_batch = False

            # feed_dict = {input_var: X_train_a, target_var: Y_train_a, learning_rate:lr}
            # feed_dict.update(decoded.all_drop )    # enable dropout or dropconnect layers
            # _, aug_loss = sess.run([train_op, train_cost], feed_dict=feed_dict)
            aug_train_loss = (batched_nums * aug_train_loss + _loss * X_train_a.shape[0]) / (batched_nums + X_train_a.shape[0])
            batched_nums = batched_nums + X_train_a.shape[0]
            # print(aug_train_loss)
        aug_train_loss = hvd.allreduce(aug_train_loss)
        if hvd.rank() == 0:
            print("{}/{}  {:.3f}s  loss : {:.3f}".format(epoch + 1, nb_epoch, time.time() - start_time, aug_train_loss))

        # if epoch + 1 == 1:
        #     train_log_infos = np.array([[epoch + 1, aug_train_loss]])
        # else:
        #     train_log_infos = np.append(train_log_infos, [[epoch + 1, aug_train_loss]], axis=0)

        # try to use all GPUs for inference
        if (not False) and (epoch + 1 == 1 or (epoch + 1) % print_freq == 0):
            train_loss = 0
            train_acc = 0
            train_reconst_err_1 = 0
            train_reconst_err_2 = 0
            train_precision = 0
            train_recall = 0
            train_f1_score = 0
            batched_nums = 0

            for X_train_a, Y_train_a in train_inf_ds:
                prediction_a = network(X_train_a, is_train=False)
                loss, acc, err_1, err_2, precision, recall, f1_score = performance_criterion_function(Y_train_a,
                                                                                                      prediction_a,
                                                                                                      intensity_thrsh=intensity_thrsh,
                                                                                                      loss_func_handle=loss_func_handle)

                train_loss = (batched_nums * train_loss + loss * X_train_a.shape[0]) / (
                            batched_nums + X_train_a.shape[0])
                train_acc = (batched_nums * train_acc + acc * X_train_a.shape[0]) / (batched_nums + X_train_a.shape[0])
                train_reconst_err_1 = (batched_nums * train_reconst_err_1 + err_1 * X_train_a.shape[0]) / \
                                      (batched_nums + X_train_a.shape[0])
                train_reconst_err_2 = (batched_nums * train_reconst_err_2 + err_2 * X_train_a.shape[0]) / \
                                      (batched_nums + X_train_a.shape[0])
                train_precision = (batched_nums * train_precision + precision * X_train_a.shape[0]) / \
                                  (batched_nums + X_train_a.shape[0])
                train_recall = (batched_nums * train_recall + recall * X_train_a.shape[0]) / \
                               (batched_nums + X_train_a.shape[0])
                train_f1_score = (batched_nums * train_f1_score + f1_score * X_train_a.shape[0]) / \
                                 (batched_nums + X_train_a.shape[0])
                batched_nums += X_train_a.shape[0]

            train_loss = hvd.allreduce(train_loss)
            train_acc = hvd.allreduce(train_acc)
            train_reconst_err_1 = hvd.allreduce(train_reconst_err_1)
            train_reconst_err_2 = hvd.allreduce(train_reconst_err_2)
            train_precision = hvd.allreduce(train_precision)
            train_recall = hvd.allreduce(train_recall)
            train_f1_score = hvd.allreduce(train_f1_score)

            if hvd.rank() == 0:
                # print("  training augment loss:\t{:.6f}".format(train_aug_loss))
                print("  training loss:\t\t{:.6f}".format(train_loss))
                print("  training accuracy:\t\t{:.2f} %".format(train_acc * 100))
                print("  training error:\t\t{:.2f} %".format(train_reconst_err_1 * 100))
                print("  training error (JI):\t\t{:.2f} %".format(train_reconst_err_2 * 100))
                print("  training precision:\t\t{:.2f} %".format(train_precision * 100))
                print("  training recall:\t\t{:.2f} %".format(train_recall * 100))
                print("  training f1 score:\t\t{:.2f} %".format(train_f1_score * 100))

            batched_nums = 0
            test_loss = 0
            test_acc = 0
            test_reconst_err_1 = 0
            test_reconst_err_2 = 0
            test_precision = 0
            test_recall = 0
            test_f1_score = 0
            for X_test_a, Y_test_a in test_ds:
                prediction_a = network(X_test_a, is_train=False)
                loss, acc, err_1, err_2, precision, recall, f1_score = performance_criterion_function(Y_test_a,
                                                                                                      prediction_a,
                                                                                                      intensity_thrsh=intensity_thrsh,
                                                                                                      loss_func_handle=loss_func_handle)
                test_loss = (batched_nums * test_loss + loss * X_test_a.shape[0]) / (batched_nums + X_test_a.shape[0])
                test_acc = (batched_nums * test_acc + acc * X_test_a.shape[0]) / (batched_nums + X_test_a.shape[0])
                test_reconst_err_1 = (batched_nums * test_reconst_err_1 + err_1 * X_test_a.shape[0]) / \
                                     (batched_nums + X_test_a.shape[0])
                test_reconst_err_2 = (batched_nums * test_reconst_err_2 + err_2 * X_test_a.shape[0]) / \
                                     (batched_nums + X_test_a.shape[0])
                test_precision = (batched_nums * test_precision + precision * X_test_a.shape[0]) / \
                                 (batched_nums + X_test_a.shape[0])
                test_recall = (batched_nums * test_recall + recall * X_test_a.shape[0]) / \
                              (batched_nums + X_test_a.shape[0])
                test_f1_score = (batched_nums * test_f1_score + f1_score * X_test_a.shape[0]) / \
                                (batched_nums + X_test_a.shape[0])
                batched_nums += X_test_a.shape[0]

            test_loss = hvd.allreduce(test_loss)
            test_acc = hvd.allreduce(test_acc)
            test_reconst_err_1 = hvd.allreduce(test_reconst_err_1)
            test_reconst_err_2 = hvd.allreduce(test_reconst_err_2)
            test_precision = hvd.allreduce(test_precision)
            test_recall = hvd.allreduce(test_recall)
            test_f1_score = hvd.allreduce(test_f1_score)

            if hvd.rank() == 0 and (epoch + 1 == 1 or (epoch + 1) % print_freq == 0):
                print("  validation loss:\t\t{:.6f}".format(test_loss))
                print("  validation accuracy:\t\t{:.2f} %".format(test_acc * 100))
                print("  validation error:\t\t{:.2f} %".format(test_reconst_err_1 * 100))
                print("  validation error (JI):\t{:.2f} %".format(test_reconst_err_2 * 100))
                print("  validation precision:\t\t{:.2f} %".format(test_precision * 100))
                print("  validation recall:\t\t{:.2f} %".format(test_recall * 100))
                print("  validation f1 score:\t\t{:.2f} %".format(test_f1_score * 100))

            # detemine whether current error is smaller than stored errors
            if epoch >= nb_epoch - 100 and hvd.rank() == 0 and np.any(train_reconst_err_1 < best_train_error_mat):
                max_ind = np.argmax(best_train_error_mat)
                # print(best_train_error_mat[max_ind], best_model_inds[max_ind])
                model_to_delete = "%smodel_epoch_%d%s.h5" % (model_dir, best_model_inds[max_ind], model_suffix)
                try:
                    os.remove(model_to_delete)
                except:
                    pass

                best_train_error_mat[max_ind] = train_reconst_err_1
                best_model_inds[max_ind] = epoch
                model_to_save = "%smodel_epoch_%d%s.h5" % (model_dir, epoch, model_suffix)
                network.save_weights(model_to_save)

            if hvd.rank() == 0:
                if epoch + 1 == 1:
                    log_infos = np.array([[epoch + 1, train_loss, train_acc, train_reconst_err_1, train_reconst_err_2,
                                           train_precision, train_recall, train_f1_score, test_loss, test_acc,
                                           test_reconst_err_1, test_reconst_err_2, test_precision, test_recall,
                                           test_f1_score]])
                else:
                    log_infos = np.append(log_infos,
                                          [[epoch + 1, train_loss, train_acc, train_reconst_err_1, train_reconst_err_2,
                                            train_precision, train_recall, train_f1_score, test_loss, test_acc,
                                            test_reconst_err_1,
                                            test_reconst_err_2, test_precision, test_recall, test_f1_score]], axis=0)

    if hvd.rank() == 0:
        # save the model
        # saver = tf.compat.v1.train.Saver()
        save_path = "%smodel_final_epoch_%s.h5" % (model_dir, model_suffix)
        network.save_weights(save_path)
        # save model using npz format
        # tl.files.save_npz(decoded.all_params , name="%smodel%s.npz" % (model_dir, model_suffix))

        # load the best model and perform reconstruction
        best_model_filename = model_to_save
        network.load_weights(best_model_filename)
        print("Best model restored!")

        # train_encoded_info = np.array([])
        train_decoded_image = np.array([])
        # test_encoded_info = np.array([])
        test_decoded_image = np.array([], dtype='bool_')
        if not False:
            print("Training image inference...")
            for X_train_a, inds_a in iterate_minibatches_single_data(X_train, batch_size, shuffle=False,
                                                                     throw_last_incomplete_batch=False):
                if X_train_a.shape[0] < batch_size:
                    X_train_pad = np.zeros(shape=(batch_size - X_train_a.shape[0],) + X_train_a.shape[1:])
                    X_train_a = np.concatenate((X_train_a, X_train_pad), axis=0)
                prediction_a = network(X_train_a.astype(np.float32), is_train=False)
                if len(train_decoded_image.shape) == 1:
                    train_decoded_image = prediction_a.numpy()
                else:
                    train_decoded_image = np.concatenate((train_decoded_image, prediction_a.numpy()), axis=0)
            train_decoded_image = train_decoded_image[:X_train.shape[0], ]
            print("Training image inference finished!")
        del X_train

        print("Testing image inference...")
        for X_test_a, inds_a in iterate_minibatches_single_data(X_test, batch_size, shuffle=False,
                                                                throw_last_incomplete_batch=False):
            if X_test_a.shape[0] < batch_size:
                X_test_pad = np.zeros(shape=(batch_size - X_test_a.shape[0],) + X_test_a.shape[1:])
                X_test_a = np.concatenate((X_test_a, X_test_pad), axis=0)
            prediction_a = network(X_test_a.astype(np.float32), is_train=False)
            if len(test_decoded_image.shape) == 1:
                test_decoded_image = prediction_a.numpy()
            else:
                test_decoded_image = np.concatenate((test_decoded_image, prediction_a.numpy()), axis=0)
            test_decoded_image = test_decoded_image[:X_test.shape[0], ]
        print("Testing image inference finished!")

        encode_reconst_train_filename = "%s/deep_conv_ae_encode_reconstruction_train_results_ld_%d%s.mat" % (
        image_dir, latent_dim, model_suffix)
        encode_reconst_test_filename = "%s/deep_conv_ae_encode_reconstruction_test_results_ld_%d%s.mat" % (
        image_dir, latent_dim, model_suffix)
        hdf5storage.savemat(encode_reconst_train_filename, {'train_decoded_image': train_decoded_image,
                                                            'best_train_error_mat': best_train_error_mat,
                                                            'best_model_inds': best_model_inds,
                                                            'best_model_filename': best_model_filename,
                                                            'log_infos': log_infos}, format='7.3', oned_as='column',
                            store_python_metadata=True)
        hdf5storage.savemat(encode_reconst_test_filename,
                            {'test_decoded_image': test_decoded_image, 'log_infos': log_infos},
                            format='7.3', oned_as='column', store_python_metadata=True)
        return log_infos

    return 1


def main():
    multiple_running = True
    if multiple_running:
        run_set_list = ['test_methods', 'test_methods_1', 'test_methods_2', 'test_methods_3', 'test_methods_4',
                        'test_methods_5', 'test_methods_6', 'test_methods_7']
        if len(sys.argv) > 1 and sys.argv[1] in run_set_list:
            run_set = sys.argv[1]
        else:
            run_set = run_set_list[0]
        print('run set: ', run_set)
        if run_set == '':
            pass
        elif run_set == 'test_methods':
            pass
        elif run_set == 'test_methods_1':
            latent_dims = [500]
            layer_types = ['FC']
            data_types = ['snr-3']
            data_options = ['size_1024']
            test_param_list = [[1], [2], [4], [8], [16], [32], [64], [128], ['cell']]
            test_param_names = ['filter_n1', 'filter_n2', 'filter_n3', 'filter_n4', 'filter_n5', 'filter_n6',
                                'filter_n7', 'filter_n8', 'cell_component']
            test_params = dict(zip(test_param_names, test_param_list))
            num_run = 1
            result_directory = '../../results/test_HPA_deep_ae_v1_rc1_cell_sz_512_test_methods_oldnet_augment_rlr_2_08162019001/'
        elif run_set == 'test_methods_2':
            latent_dims = [500]
            layer_types = ['FC']
            data_types = ['snr-2']
            data_options = ['size_1024']
            test_param_list = [[1], [2], [4], [8], [16], [32], [64], [128], ['cell']]
            test_param_names = ['filter_n1', 'filter_n2', 'filter_n3', 'filter_n4', 'filter_n5', 'filter_n6',
                                'filter_n7', 'filter_n8', 'cell_component']
            test_params = dict(zip(test_param_names, test_param_list))
            num_run = 1
            result_directory = '../../results/test_point_detection_u_net_augment_snr_2_08212019001/'
        elif run_set == 'test_methods_3':
            latent_dims = [500]
            layer_types = ['FC']
            data_types = ['snr-2']
            data_options = ['size_1024']
            test_param_list = [[1], [2], [4], [8], [16], [32], [64], [128], ['cell']]
            test_param_names = ['filter_n1', 'filter_n2', 'filter_n3', 'filter_n4', 'filter_n5', 'filter_n6',
                                'filter_n7', 'filter_n8', 'cell_component']
            test_params = dict(zip(test_param_names, test_param_list))
            num_run = 1
            result_directory = '../../results/test_point_detection_u_net_augment_snr_1.5_08212019001/'
        elif run_set == 'test_methods_4':
            latent_dims = [500]
            layer_types = ['FC']
            data_types = ['snr-3']
            data_options = ['size_512']
            test_param_list = [[1], [2], [4], [8], [16], [32], [64], [128], ['cell'], [24000], [512], [8]]
            test_param_names = ['filter_n1', 'filter_n2', 'filter_n3', 'filter_n4', 'filter_n5', 'filter_n6',
                                'filter_n7', 'filter_n8', 'cell_component', 'train_num', 'image_size', 'batch_size']
            test_params = dict(zip(test_param_names, test_param_list))
            num_run = 1
            result_directory = '../../results/test_point_detection_u_net_augment_snr_3_08292019001/'
        elif run_set == 'test_methods_5':
            latent_dims = [500]
            layer_types = ['FC']
            data_types = ['snr-3']
            data_options = ['size_512_small']
            test_param_list = [[1], [2], [4], [8], [16], [32], [64], [128], ['cell'], [3000], [512], [8], [6000], ['dice_coe']]
            test_param_names = ['filter_n1', 'filter_n2', 'filter_n3', 'filter_n4', 'filter_n5', 'filter_n6',
                                'filter_n7', 'filter_n8', 'cell_component', 'train_num', 'image_size', 'batch_size',
                                'nb_epoch', 'loss_func']
            test_params = dict(zip(test_param_names, test_param_list))
            num_run = 1
            result_directory = '../../results/test_point_detection_u_net_augment_snr_3_08292019001/'
            result_directory = '../../results/test_point_detection_u_net_augment_snr_3_08292019002/'
            result_directory = '../../results/test_point_detection_u_net_augment_snr_3_09012019001/'  # decrease lr
            result_directory = '../../results/test_point_detection_u_net_augment_snr_3_09022019001/'  # batch size 4
            result_directory = '../../results/test_point_detection_u_net_augment_snr_3_09032019001/'  # batch size 8, train size 3000, 3000 epoches
            result_directory = '../../results/test_point_detection_u_net_augment_snr_3_09062019001/'  # batch size 8, train size 3000, 6000 epoches
        elif run_set == 'test_methods_6':
            latent_dims = [500]
            layer_types = ['FC']
            data_types = ['snr-3']
            data_options = ['size_512_small']
            test_param_list = [[1], [2], [4], [8], [16], [32], [64], [128], ['cell'], [3000], [512], [8], [6000], ['hinge']]
            test_param_names = ['filter_n1', 'filter_n2', 'filter_n3', 'filter_n4', 'filter_n5', 'filter_n6',
                                'filter_n7', 'filter_n8', 'cell_component', 'train_num', 'image_size', 'batch_size',
                                'nb_epoch', 'loss_func']
            test_params = dict(zip(test_param_names, test_param_list))
            num_run = 1
            result_directory = '../../results/test_point_detection_u_net_augment_snr_3_09062019001/'  # batch size 8, train size 3000, 6000 epoches
        elif run_set == 'test_methods_7':
            latent_dims = [500]
            layer_types = ['FC']
            data_types = ['snr-2']
            data_options = ['size_512_small']
            test_param_list = [[1], [2], [4], [8], [16], [32], [64], [128], ['cell'], [6000], [512], [8], [3000], ['dice_coe']]
            test_param_names = ['filter_n1', 'filter_n2', 'filter_n3', 'filter_n4', 'filter_n5', 'filter_n6',
                                'filter_n7', 'filter_n8', 'cell_component', 'train_num', 'image_size', 'batch_size',
                                'nb_epoch', 'loss_func']
            test_params = dict(zip(test_param_names, test_param_list))
            num_run = 1
            result_directory = '../../results/test_point_detection_u_net_augment_snr_3_09102019001/'  # batch size 8, train size 3000, 6000 epoches
            result_directory = '../../results/test_point_detection_u_net_augment_snr_3_09122019001/'  # batch size 8, train size 3000, 6000 epoches

        print(result_directory)

        if not os.path.exists(result_directory):
            try:
                os.makedirs(result_directory)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

        Chunk_lock_clean(result_directory, 8 * 60, True)
        for i in range(len(latent_dims)):
            latent_dim_i = latent_dims[i]

            for layer_ind in range(len(layer_types)):
                layer_type = layer_types[layer_ind]
                for data_opt_ind in range(len(data_options)):
                    data_option = data_options[data_opt_ind]
                    for data_ind in range(len(data_types)):
                        data_type = data_types[data_ind]

                        if data_option == 'size_1024':
                            image_directory = '../../results/synthetic_particle_2d_images_datasets_dl_08132019/'
                        elif data_option == 'size_512':
                            image_directory = '../../results/synthetic_particle_2d_images_datasets_dl_08262019/'
                        elif data_option == 'size_512_small':
                            image_directory = '../../results/synthetic_particle_2d_images_datasets_dl_08262019/'

                        if 'small' in data_option:
                            if data_type == 'snr-1.5':
                                data_filename = image_directory + 'point_source_synthetic_2d_image_dataset_7500_snr_1.5.mat'
                            elif data_type == 'snr-2':
                                data_filename = image_directory + 'point_source_synthetic_2d_image_dataset_7500_snr_2.mat'
                            elif data_type == 'snr-3':
                                data_filename = image_directory + 'point_source_synthetic_2d_image_dataset_7500_snr_3.mat'
                        else:
                            if data_type == 'snr-1.5':
                                data_filename = image_directory + 'point_source_synthetic_2d_image_dataset_snr_1.5.mat'
                            elif data_type == 'snr-2':
                                data_filename = image_directory + 'point_source_synthetic_2d_image_dataset_snr_2.mat'
                            elif data_type == 'snr-3':
                                data_filename = image_directory + 'point_source_synthetic_2d_image_dataset_snr_3.mat'

                        N_test_params = [len(_) for _ in test_param_list]
                        N_test = np.prod(N_test_params)

                        computing_param = dict()
                        for test_ind in range(N_test):
                            Chunk_lock_clean(result_directory, 10 * 60)
                            sub_i = np.unravel_index(test_ind, N_test_params, 'F')
                            for k, cur_name in enumerate(test_param_names):
                                computing_param[cur_name] = test_param_list[k][sub_i[k]]

                            for j in range(num_run):
                                print('result_param_ld_{:d}_layer_{:s}_run_{:d}'.format(latent_dim_i, layer_type, j))
                                print('latent dim {:d}'.format(latent_dim_i))
                                print('data option {:s}'.format(data_option))
                                print('data type {:s}'.format(data_type))

                                cur_filename = '%sresult_param_ld_%d_layer_%s_data_%s_option_%s_param_%d_run_%d.mat' % (
                                result_directory, latent_dim_i, layer_type, data_type, data_option, test_ind, j)
                                cur_tmpname = '%sresult_param_ld_%d_layer_%s_data_%s_option_%s_param_%d_run_%d.tmp' % (
                                result_directory, latent_dim_i, layer_type, data_type, data_option, test_ind, j)
                                if os.path.exists(cur_filename):
                                    print('the computing for this time has finished, skip it!')
                                    # continue
                                if os.path.exists(cur_tmpname):
                                    print('the computing for this time is being doing by another job, skip it!')
                                    # continue
                                fid = open(cur_tmpname, 'w')
                                fid.close

                                default_computing_param = {'filter_n1': 1, 'filter_n2': 2, 'filter_n3': 4,
                                                           'filter_n4': 8, 'filter_n5': 16, 'filter_n6': 32,
                                                           'filter_n7': 64, 'filter_n8': 128, 'batch_size': 64,
                                                           'nb_epoch': 2000, 'train_num': 6000,
                                                           'lr_method': 'step', 'cell_component': 'cell',
                                                           'do_augmentation': True, 'image_size': 1024,
                                                           'loss_func': 'binary_cross_entropy'
                                                           }
                                computing_param = process_options_structure(default_computing_param, computing_param)

                                print(computing_param)
                                filter_n1 = computing_param['filter_n1']
                                filter_n2 = computing_param['filter_n2']
                                filter_n3 = computing_param['filter_n3']
                                filter_n4 = computing_param['filter_n4']
                                filter_n5 = computing_param['filter_n5']
                                filter_n6 = computing_param['filter_n6']
                                filter_n7 = computing_param['filter_n7']
                                filter_n8 = computing_param['filter_n8']

                                nb_epoch = computing_param['nb_epoch']
                                train_num = computing_param['train_num']
                                image_size = computing_param['image_size']
                                lr_method = computing_param['lr_method']
                                cell_component = computing_param['cell_component']
                                do_augmentation = computing_param['do_augmentation']
                                batch_size = computing_param['batch_size']
                                loss_func = computing_param['loss_func']

                                model_suffix = '_ld_%d_layer_%s_data_%s_option_%s_param_%d_run_%d' % (
                                latent_dim_i, layer_type, data_type, data_option, test_ind, j)
                                print(model_suffix)
                                log_infos = deep_conv_autoencoder(latent_dim=latent_dim_i, model_suffix=model_suffix,
                                                                  data_filename=data_filename,
                                                                  filter_n1=filter_n1, filter_n2=filter_n2,
                                                                  filter_n3=filter_n3, filter_n4=filter_n4,
                                                                  filter_n5=filter_n5, filter_n6=filter_n6,
                                                                  filter_n7=filter_n7, filter_n8=filter_n8,
                                                                  nb_epoch=nb_epoch, train_num=train_num,
                                                                  lr_method=lr_method, data_option=data_option,
                                                                  data_type=data_type, cell_component=cell_component,
                                                                  do_augmentation=do_augmentation, batch_size=batch_size,
                                                                  layer_type=layer_type, image_size=image_size,
                                                                  loss_func=loss_func,
                                                                  result_directory=result_directory)
                                # tl.layers.clear_layers_name()
                                tf.compat.v1.reset_default_graph()

                                sio.savemat(cur_filename, {'latent_dim': latent_dim_i, 'layer_type': layer_type,
                                                           'data_type': data_type, 'computing_param': computing_param,
                                                           'log_infos': log_infos})

                                if os.path.exists(cur_tmpname):
                                    os.remove(cur_tmpname)
    else:
        pass


if __name__ == '__main__':
    main()

