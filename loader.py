
from keras.utils import np_utils
import numpy as np
import pandas as pd
import plot


def process_data(audio_data):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    features = np.array(audio_data['Feature'].tolist())
    labels = np.array(audio_data['Label'].tolist())

    lab_encode = LabelEncoder()

    labels_cate = np_utils.to_categorical(lab_encode.fit_transform(labels))

    x_train, x_test, y_train, y_test = train_test_split(features, labels_cate,
                                                        test_size=0.20)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    return x_train, x_test, y_train, y_test


def get_params():
    params = {}

    # General Convolutional Layer Parameters
    params['stride'] = 1
    params['cnn_activate'] = 'relu'

    # Convolutional Layer Parameters
    params['filters_1'] = 20
    params['kernel_1'] = 4

    # Compile Parameters
    params['optimizer'] = 'adam'
    params['loss'] = 'categorical_crossentropy'

    # Dense Layer Parameters
    params['dense_1'] = 120
    params['activate_1'] = 'relu'  # Used for CNN

    # Fit Parameters
    params['epoch'] = 200
    params['batch_size'] = 32
    return params


def basic_cnn(params):
    """ Builds basic Convolutional neural network model """
    from keras.layers import Dense, Flatten, MaxPooling1D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Conv1D
    from keras.models import Sequential

    model = Sequential()
    model.add(Conv1D(params['filters_1'], params['kernel_1'],
                     input_shape=(40, 1,),
                     strides=params['stride'],
                     activation=params['cnn_activate'],
                     padding='same'))
    model.add(BatchNormalization())

    model.add(Conv1D(params['filters_1'], params['kernel_1'],
                     strides=params['stride'],
                     activation=params['cnn_activate'],
                     padding='same'))
    model.add(BatchNormalization())

    model.add(Conv1D(params['filters_1'], params['kernel_1'],
                     strides=params['stride'],
                     activation=params['cnn_activate'],
                     padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling1D(padding='same'))
    #model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(params['dense_1'], activation=params['activate_1']))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])

    print(model.summary())
    return model


def fit_model(model, params, x_train, y_train, x_test, y_test):
    """ Fits neural network """
    # Fits Model
    model.fit(x_train, y_train,
              epochs=params['epoch'],
              batch_size=params['batch_size'],
              verbose=1)

    # Predicts Model
    y_pred = model.predict(x_test)
    y_pred_rounded = y_pred.round()

    # Scores Model on Test Data
    metrics = {'acc': 0.0, 'loss': 0.0}
    metrics['loss'], metrics['acc'] = model.evaluate(x_test, y_test, batch_size=128)
    print('\nAccuracy {} & Loss {}\n'.format(metrics['acc'], metrics['loss']))

    return y_pred_rounded, metrics


def main():
    # Reads and Processes Data
    audio_data = pd.read_pickle('./data/set_a_parsed.pkl')
    x_train, x_test, y_train, y_test = process_data(audio_data)

    # Reads default options for model & generate list of classes
    params = get_params()
    classes = ['artifact', 'extrahls', 'murmur', 'normal']

    # Creates range to loop filter between
    opt_name = 'conv_filters'
    options = [4, 8, 12, 16, 24, 28, 32, 36]
    history_dict = {x: {'loss': 0.0, 'acc': 0.0} for x in options}
    
    # Loops between each range and tests it for 3 iterations each
    for opt in options:
        history_dict[opt] = {'loss': [], 'acc': []}
        params[opt_name] = opt
        for _ in [1, 2, 3]:
            # Creates CNN and fits the model to it
            model = basic_cnn(params)
            y_pred, metrics = fit_model(model, params, x_train, y_train,
                                        x_test, y_test)

            # Adds Data to Trends
            history_dict[opt]['loss'].append(metrics['loss'])
            history_dict[opt]['acc'].append(metrics['acc'])
        # Exit For Loop for Iterations

        # Calculates Average
        history_dict[opt]['loss'] = np.mean(history_dict[opt]['loss'])
        history_dict[opt]['acc'] = np.mean(history_dict[opt]['acc'])

        # Plots Confusion Matrix
        title = "{} (Loss {} & Acc {})".format(opt, metrics['loss'], metrics['acc'])
        conf_png = './tempplots/{}_{}.png'.format(opt, opt_name)
        plot.conf_matrix(y_test, y_pred, classes, out=conf_png, title=title)
    # Exit For Loop for each Range


if __name__ == "__main__":
    main()
