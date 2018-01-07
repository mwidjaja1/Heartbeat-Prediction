#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 13:21:50 2017

@author: matthew
"""

def standard():
    params = {}

    # Build Convolutional Layers
    params['conv_filters'] = 20
    params['kernel_size'] = 4
    params['kernel_stride'] = 1
    params['cnn_activate'] = 'relu'

    # Fit Parameters
    params['optimizer'] = 'adam'
    params['loss'] = 'categorical_crossentropy'

    # Dense Layer
    params['dense_1'] = 120
    # params['activate_1'] = 'relu'  # Used for CNN
    params['activate_1'] = 'sigmoid'  # Used for Neural

    # VGG Parameters
    params['vgg_filters_1'] = 8
    params['vgg_filters_2'] = 16
    params['vgg_filters_3'] = 32
    params['vgg_filters_4'] = 64
    params['vgg_kernel'] = 4
    params['vgg_dense'] = 504
    params['vgg_activation'] = 'relu'

    # ResNet Parameters
    params['res_filters_1'] = 4
    params['res_filters_3'] = 8
    params['res_filters_4'] = 16
    params['res_filters_5'] = 32
    params['res_activate'] = 'relu'
    params['res_kernel_size'] = 4
    params['res_dense'] = 504

    return params
    