
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def load_csv(csv_path):
    data = pd.read_csv(csv_path)
    data_use = data[pd.notnull(data['label'])]
    return zip(data_use.fname, data_use.label)


def parse_single_audio(wav_rel_path):
    import librosa
    import librosa.display

    wav_path = './data/{}'.format(wav_rel_path)
    hb_name = os.path.basename(wav_rel_path)

    data, sample_rate = librosa.load(wav_path)
    feature = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T,
                      axis=0)

    """
    plt.figure()
    librosa.display.waveplot(data, sr=sample_rate)
    plt.savefig('plots/{}.png'.format(hb_name))
    """
    return feature


def process_data(audio_data):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    features = np.array(audio_data['Feature'])
    labels = np.array(audio_data['Label'])
    
    lb = LabelEncoder()
    
    labels_cate = np_utils.to_categorical(lb.fit_transform(labels))

    x_train, x_test, y_train, y_test = train_test_split(features, labels_cate,
                                                        test_size=0.20)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    return x_train, x_test, y_train, y_test


def cnn(label_size):
    filter_size = 2

    model = Sequential()

    model.add(Dense(256, input_shape=(40,), activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(label_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'], optimizer='adam')
    return model


def basic_cnn(shape):
    """ Builds basic Convolutional neural network model """
    from keras.layers import Dense, Dropout, Flatten, InputLayer, MaxPooling1D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Conv1D
    from keras.models import Sequential

    model = Sequential()

    model.add(Conv1D(20,
                     2,
                     input_shape=(40, 1,),
                     strides=1,
                     padding='same'))
    model.add(MaxPooling1D(padding='same'))
    model.add(Dropout(0.1))
    model.add(Flatten())

    model.add(Dense(120, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())
    return model


def main():
    audio_files = load_csv('./data/set_a.csv')
    audio_data = {'Feature': [], 'Label': []}

    for wav_rel_path, hb_type in audio_files:
        try:
            feature = parse_single_audio(wav_rel_path)
            audio_data['Feature'].append(feature)
            audio_data['Label'].append(hb_type)
        except Exception as err:
            print(err)

    x_train, x_test, y_train, y_test = process_data(audio_data)

    model = basic_cnn(x_train.shape)
    model.fit(x_train, y_train, batch_size=32, epochs=20,
              validation_data=(x_test, y_test))


if __name__ == "__main__":
    main()