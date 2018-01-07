
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd



def make_array(text):
    return np.fromstring(text, sep='   ')


def process_data(audio_data):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    features = np.array(audio_data['Feature'].tolist())
    labels = np.array(audio_data['Label'].tolist())
    
    lb = LabelEncoder()
    
    labels_cate = np_utils.to_categorical(lb.fit_transform(labels))

    x_train, x_test, y_train, y_test = train_test_split(features, labels_cate,
                                                        test_size=0.20)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    return x_train, x_test, y_train, y_test


def basic_cnn(shape):
    """ Builds basic Convolutional neural network model """
    from keras.layers import Dense, Dropout, Flatten, InputLayer, MaxPooling1D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Conv1D
    from keras.models import Sequential
    import params
    
    opt = params.standard()

    model = Sequential()
    model.add(Conv1D(opt['conv_filters'], opt['kernel_size'],
                     input_shape=(40, 1,),
                     strides=opt['kernel_stride'],
                     activation=opt['cnn_activate'],
                     padding='same'))
    model.add(BatchNormalization())

    model.add(Conv1D(opt['conv_filters'], opt['kernel_size'],
                     strides=opt['kernel_stride'],
                     activation=opt['cnn_activate'],
                     padding='same'))
    model.add(BatchNormalization())

    model.add(Conv1D(opt['conv_filters'], opt['kernel_size'],
                     strides=opt['kernel_stride'],
                     activation=opt['cnn_activate'],
                     padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling1D(padding='same'))
    #model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(opt['dense_1'], activation=opt['activate_1']))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss=opt['loss'],
                  optimizer=opt['optimizer'],
                  metrics=['accuracy'])

    print(model.summary())
    return model


def main():
    # Reads and Processes Data
    audio_data = pd.read_pickle('./data/set_a_parsed.pkl')
    x_train, x_test, y_train, y_test = process_data(audio_data)

    # Creates range to loop filter between
    change = 'gpu_triplecnn_batch'
    range = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    history_dict = {x: {'loss': 0.0, 'acc': 0.0} for x in range}

    model = basic_cnn(x_train.shape)
    model.fit(x_train, y_train, epochs=100,
              validation_data=(x_test, y_test))


if __name__ == "__main__":
    main()