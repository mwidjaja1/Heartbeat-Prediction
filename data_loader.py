import numpy as np
import os
import pandas as pd

def load_csv(csv_path):
    """ Opens up CSV file which contains the data paths for each audio file
        & their heartbeat type. We save the paths & each type as a tuple.
    """
    data = pd.read_csv(csv_path)
    data_use = data[pd.notnull(data['label'])]
    return zip(data_use.fname, data_use.label)


def parse_single_audio(wav_rel_path):
    """ Parses a single audio file to extract key features from it
        and outputs the array of these features.
    """
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


def main():
    """ Parses the CSV file and the audio files it links to. We extract
        each heartbeat's key features and its type, save it to a Pandas
        DataFrame, which is then saved as another Pickle file.
    """
    audio_files = load_csv('./data/set_a.csv')
    audio_data = {'Feature': [], 'Label': []}

    for wav_rel_path, hb_type in audio_files:
        try:
            feature = parse_single_audio(wav_rel_path)
            audio_data['Feature'].append(feature)
            audio_data['Label'].append(hb_type)
        except Exception as err:
            print(err)

    audio_df = pd.DataFrame.from_dict(audio_data)
    audio_df.to_pickle('./data/set_a_parsed.pkl')


if __name__ == "__main__":
    main()
