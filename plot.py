#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 21:41:41 2017

@author: matthew
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def conf_matrix(y_test, y_test_predict, classes, title='Confusion Matrix',
                out=None):
    # Converts both output arrays into just one column based on the class
    y_test_predict_class = y_test_predict.argmax(1)
    y_test_class = y_test.argmax(1)

    # Creates confusion matrix
    cm_data = confusion_matrix(y_test_class, y_test_predict_class)
    np.set_printoptions(precision=2)

    # Plots Confusion Matrix
    plt.figure()
    plt.imshow(cm_data, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.xlabel('Predicted Label')
    plt.yticks(tick_marks, classes)
    plt.ylabel('True Label')

    # Plots data on chart
    thresh = cm_data.max() / 2.
    for i, j in itertools.product(range(cm_data.shape[0]), range(cm_data.shape[1])):
        plt.text(j, i, format(cm_data[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_data[i, j] > thresh else "black")

    plt.tight_layout()

    # Saves or Shows Plot
    if out:
        plt.savefig(out)
    else:
        plt.show()