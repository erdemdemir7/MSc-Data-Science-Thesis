# -*- coding: utf-8 -*-
import logging
import os
import pickle
import sys
import time
import warnings
import math
from argparse import ArgumentParser
from pprint import pformat, pprint

from tfsnippet.utils import makedirs

import numpy as np
import tensorflow as tf
from tfsnippet.examples.utils import MLResults, print_with_title
from tfsnippet.scaffold import VariableSaver
from tfsnippet.utils import get_variables_as_dict, register_config_arguments, Config

from omni_anomaly.eval_methods import pot_eval, bf_search
from omni_anomaly.model import OmniAnomaly
from omni_anomaly.prediction import Predictor
from omni_anomaly.training import Trainer
from omni_anomaly.utils import get_data_dim, get_data, save_z


class ExpConfig(Config):
    # dataset configuration
    dataset = "machine-1-1"
    testOn1 = 'SMD'
    testOn2 = 'MSL'
    x_dim = 25

    # model architecture configuration
    use_connected_z_q = True
    use_connected_z_p = True

    # model parameters
    z_dim = 3
    # z_dim = 3
    rnn_cell = 'GRU'  # 'GRU', 'LSTM' or 'Basic'
    window_length = 400
    rnn_num_hidden = 64
    # rnn_num_hidden = 500
    dense_dim = 64
    # dense_dim = 500
    posterior_flow_type = 'nf'  # 'nf' or None
    nf_layers = 20  # for nf
    # nf_layers = 20  # for nf
    max_epoch = 3
    # max_epoch = 5
    train_start = 0
    max_train_size = None  # `None` means full train set
    # batch_size = 1
    batch_size = 1

    l2_reg = 0.0001
    initial_lr = 0.0001
    # initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 40
    lr_anneal_step_freq = None
    std_epsilon = 1e-4

    # evaluation parameters
    test_n_z = 1
    test_batch_size = 1
    test_start = 0
    max_test_size = None  # `None` means full test set

    # the range and step-size for score for searching best-f1
    # may vary for different dataset
    bf_search_min = -400.
    bf_search_max = 400.
    bf_search_step_size = 1.

    valid_step_freq = 100
    gradient_clip_norm = 10.

    early_stop = True  # whether to apply early stop method

    # pot parameters
    # recommend values for `level`:
    # SMAP: 0.07
    # MSL: 0.01
    # SMD group 1: 0.0050
    # SMD group 2: 0.0075
    # SMD group 3: 0.0001
    level = 0.07


    # outputs config
    save_z = False  # whether to save sampled z in hidden space
    get_score_on_dim = False  # whether to get score on dim. If `True`, the score will be a 2-dim ndarray
    save_dir = 'model'
    restore_dir = None  # If not None, restore variables from this dir
    result_dir = 'result'  # Where to save the result file

    test_score_filename1 = testOn1 + '_test_score.pkl'
    test_score_filename2 = testOn2 + '_test_score.pkl'


def main():
    import os

    os.environ["PYTHONIOENCODING"] = "utf8"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    logging.basicConfig(
        level='INFO',
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # prepare test data
    (_, _), (x_test1, y_test1) = \
        get_data(config.testOn1, config.max_train_size, config.max_test_size, train_start=config.train_start,
                 test_start=config.test_start)

    # prepare test data
    (_, _), (x_test2, y_test2) = \
        get_data(config.testOn2, config.max_train_size, config.max_test_size, train_start=config.train_start,
                 test_start=config.test_start)


    # construct the model under `variable_scope` named 'model'
    with tf.variable_scope('model') as model_vs:
        model = OmniAnomaly(config=config, name="model")

        # construct the trainer
        trainer = Trainer(model=model,
                          model_vs=model_vs,
                          max_epoch=config.max_epoch,
                          batch_size=config.batch_size,
                          valid_batch_size=config.test_batch_size,
                          initial_lr=config.initial_lr,
                          lr_anneal_epochs=config.lr_anneal_epoch_freq,
                          lr_anneal_factor=config.lr_anneal_factor,
                          grad_clip_norm=config.gradient_clip_norm,
                          valid_step_freq=config.valid_step_freq,
                          config=config)

        # construct the predictor
        predictor = Predictor(model, batch_size=config.batch_size, n_z=config.test_n_z,
                              last_point_only=True)

        with tf.Session().as_default():

            if config.restore_dir is not None:
                # Restore variables from `restore_dir`.
                saver = VariableSaver(get_variables_as_dict(model_vs), config.restore_dir)
                saver.restore()

            # if config.max_epoch > 0:
            #     # train the model
            #     train_start = time.time()
            #     best_valid_metrics = trainer.fit(x_train)
            #     train_time = (time.time() - train_start) / config.max_epoch
            #     best_valid_metrics.update({
            #         'train_time': train_time
            #     })
            # else:
                best_valid_metrics = {}

            # # get score of train set for POT algorithm
            # train_score, train_z, train_pred_speed = predictor.get_score(x_train, True)
            # if config.train_score_filename is not None:
            #     with open(os.path.join(config.result_dir, config.train_score_filename), 'wb') as file:
            #         pickle.dump(train_score, file)
            # if config.save_z:
            #     save_z(train_z, 'train_z')

            # testing
            if x_test1 is not None:
                # get score of test set
                test_start = time.time()
                test_score1, test_z1, pred_speed1 = predictor.get_score(x_test1, False)
                test_time1 = time.time() - test_start

                best_valid_metrics.update({
                    'testOn1' : config.testOn1,
                    'pred_time': pred_speed1,
                    'pred_total_time1': test_time1
                })
                if config.test_score_filename1 is not None:
                    with open(os.path.join(config.result_dir, config.test_score_filename1), 'wb') as file:
                        pickle.dump(test_score1, file)
                
                if y_test1 is not None and len(y_test1) >= len(test_score1):
                    if config.get_score_on_dim:
                        # get the joint score
                        test_score1 = np.sum(test_score, axis=-1)
                        # train_score = np.sum(train_score, axis=-1)

                    # get best f1
                    t1, th1 = bf_search(test_score1, y_test1[-len(test_score1):],
                                      start=config.bf_search_min,
                                      end=config.bf_search_max,
                                      step_num=int(abs(config.bf_search_max - config.bf_search_min) /
                                                   config.bf_search_step_size),
                                      display_freq=50)
                    # get pot results
                    # pot_result = pot_eval(train_score, test_score, y_test1[-len(test_score):], level=config.level)

                    specificity = t1[4] / (t1[4] + t1[5])
                    sensitivity = t1[3] / (t1[3] + t1[6])
                    g_mean_score = math.sqrt(specificity * sensitivity)

                    # output the results
                    best_valid_metrics.update({
                        # 'best-f1-1': t1[0],
                        # 'precision-1': t1[1],
                        'recall-1': t1[2],
                        # 'TP-1': t1[3],
                        # 'TN-1': t1[4],
                        # 'FP-1': t1[5],
                        # 'FN-1': t1[6],
                        # 'latency-1': t1[-1],
                        # 'threshold-1': th1,
                        'specificity-1' : specificity,
                        # 'sensitivity-1' : sensitivity,
                        'g_mean_score-1' : g_mean_score,
                        'Window_Size' : config.window_length,
                    })
                    # best_valid_metrics.update(pot_result)
                results.update_metrics(best_valid_metrics)

            # if config.save_dir is not None:
            #     # save the variables
            #     var_dict = get_variables_as_dict(model_vs)
            #     saver = VariableSaver(var_dict, config.save_dir)
            #     saver.save()
            # print('=' * 30 + 'result-1' + '=' * 30)
            # pprint(best_valid_metrics)

            if x_test2 is not None:
                # get score of test set
                test_start = time.time()
                test_score2, test_z2, pred_speed2 = predictor.get_score(x_test2, False)
                test_time2 = time.time() - test_start

                best_valid_metrics.update({
                    'testOn2' : config.testOn2,
                    'pred_time2': pred_speed2,
                    'pred_total_time2': test_time2
                })
                if config.test_score_filename2 is not None:
                    with open(os.path.join(config.result_dir, config.test_score_filename2), 'wb') as file:
                        pickle.dump(test_score2, file)
                
                if y_test2 is not None and len(y_test2) >= len(test_score2):
                    if config.get_score_on_dim:
                        # get the joint score
                        test_score2 = np.sum(test_score, axis=-1)
                        # train_score = np.sum(train_score, axis=-1)

                    # get best f1
                    t2, th2 = bf_search(test_score2, y_test2[-len(test_score2):],
                                      start=config.bf_search_min,
                                      end=config.bf_search_max,
                                      step_num=int(abs(config.bf_search_max - config.bf_search_min) /
                                                   config.bf_search_step_size),
                                      display_freq=50)
                    # get pot results
                    # pot_result = pot_eval(train_score, test_score, y_test1[-len(test_score):], level=config.level)

                    specificity = t2[4] / (t2[4] + t2[5])
                    sensitivity = t2[3] / (t2[3] + t2[6])
                    g_mean_score = math.sqrt(specificity * sensitivity)

                    # output the results
                    best_valid_metrics.update({
                        # 'best-f1-2': t2[0],
                        # 'precision-2': t2[1],
                        'recall-2': t2[2],
                        # 'TP-2': t2[3],
                        # 'TN-2': t2[4],
                        # 'FP-2': t2[5],
                        # 'FN-2': t2[6],
                        # 'latency-2': t2[-1],
                        # 'threshold-2': th2,
                        'specificity-2' : specificity,
                        # 'sensitivity-2' : sensitivity,
                        'g_mean_score-2' : g_mean_score,
                        'Window_Size' : config.window_length,
                    })
                    # best_valid_metrics.update(pot_result)
                results.update_metrics(best_valid_metrics)

            # if config.save_dir is not None:
            #     # save the variables
            #     var_dict = get_variables_as_dict(model_vs)
            #     saver = VariableSaver(var_dict, config.save_dir)
            #     saver.save()
            print('=' * 30 + 'result' + '=' * 30)
            pprint(best_valid_metrics)


if __name__ == '__main__':
    # get config obj
    config = ExpConfig()

    # parse the arguments
    arg_parser = ArgumentParser()
    register_config_arguments(config, arg_parser)
    arg_parser.parse_args(sys.argv[1:])
    config.x_dim = 25

    if config.dataset == 'SMAP':
        config.level = 0.07
    elif config.dataset == 'MSL':
        config.level = 0.01
    elif config.dataset == 'SMD':
        config.level = 0.0050

    folder = 'OmniAnomaly_' + config.dataset + '_' + str(config.window_length)
   
    config.save_dir = os.path.join(folder,'model/test')
    config.restore_dir = os.path.join(folder,'model')

    config.result_dir = os.path.join(folder,'result/test')  # Where to save the result file


    print_with_title('Configurations', pformat(config.to_dict()), after='\n')

    # open the result object and prepare for result directories if specified
    results = MLResults(config.result_dir)
    results.save_config(config)  # save experiment settings for review
    results.make_dirs(config.save_dir, exist_ok=True)
    with warnings.catch_warnings():
        # suppress DeprecationWarning from NumPy caused by codes in TensorFlow-Probability
        warnings.filterwarnings("ignore", category=DeprecationWarning, module='numpy')
        main()