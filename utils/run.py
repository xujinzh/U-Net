#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinzhong Xu
# @Contact : jinzhongxu@csu.ac.cn
# @Time    : 2020/10/31 11:02
# @File    : run.py
# @Software: PyCharm

from utils.model import *
from utils.augmentation import AUGMENTATION
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf
import argparse


def just_do_it():
    """
    模型训练和预测
    :return: 0
    """
    # 构造参数解析
    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--train', required=False, default='data/membrane/train/', help='path to train data set')
    ap.add_argument('-t', '--test', required=False, default='data/membrane/test/', help='path to test data set')
    ap.add_argument('-s', '--steps', required=False, type=int, default=10, help='steps per epoch for train')
    ap.add_argument('-e', '--epochs', required=False, type=int, default=5, help='epochs for train model')
    args = vars(ap.parse_args())

    # 多GPU情况下进行分布式训练
    # strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2"])
    strategy = tf.distribute.MirroredStrategy()
    data_gen_args = dict(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    aug = AUGMENTATION()
    generator = aug.train_generator(2, args['train'], 'images', 'ground_truth', data_gen_args, save_to_dir=None)
    with strategy.scope():
        model = u_net()
    model_checkpoint = ModelCheckpoint('u-net.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit(generator, steps_per_epoch=args['steps'], epochs=args['epochs'], callbacks=[model_checkpoint])

    test_generator_ = aug.test_generator(args['test'])
    results = model.predict(test_generator_, 30, verbose=1)
    aug.save_result(args['test'], results)
