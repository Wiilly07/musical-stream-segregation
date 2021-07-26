import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from voice_segregation.basic_class import Song
from voice_segregation.clustering import spectrum_system, predict_k, \
                                        minimal_overlap, pitch_prox
import setting
import numpy as np
import pandas as pd
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")

def read_midi(file, model):
    song = Song(file, reading_range=setting.reading_range)
    if setting.texture == 'polyphony':
        song.delete_same_note()

    elif setting.texture == 'homophony':
        song.delete_accompaniment()

    song.get_dis_m(model=model, default_value=0.5, reach=setting.max_dist)
    return song

def midi_save(voice, root_dir):
    f = voice.song.filename
    filename = os.path.join(root_dir, f.split('/')[-1])
    voice.output_midi(filename)

def integrated_stream(song, method, texture):
    if texture == 'polyphony':
        k = predict_k(song)
    elif texture == 'homophony':
        k = 2
    
    if method == 'pitch':
        step = setting.window - setting.overlap_pitch
        V_ls = spectrum_system(song, n=k, window=setting.window, step=step)
        V = pitch_prox(V_ls)
        return V

    elif method == 'min':
        step = setting.window - setting.overlap_min
        V_ls = spectrum_system(song, n=k, window=setting.window, step=step)
        try:
            V = minimal_overlap(V_ls)
            V.delete_same_note()
            return V
        except:
            print(f"Minimal overlap method couldn't be applied to the file {V.song.filename}, \
                please consider pitch proximity method")
            return None

def output_result(V, ls, texture):
    if V is None:
        pass
    else:
        if setting.output_midi:
            midi_save(V, root_dir=setting.output_folder)

        if texture == 'polyphony':
            metrics = [V.song.filename.split('/')[-1], V.accuracy(by='note'), V.accuracy(by='time')]
            ls.append(metrics)

        elif texture == 'homophony':
            metrics = [V.song.filename.split('/')[-1], 
                       V.precision(by='note'), V.recall(by='note'), V.f1_score(by='note'),
                       V.precision(by='time'), V.recall(by='time'), V.f1_score(by='time')]
            ls.append(metrics)


if __name__ == '__main__':
    file_list = [os.path.join(setting.input_folder, i) for i in os.listdir(setting.input_folder)]
    metrics_ls = []
    model = keras.models.load_model(setting.model_file)

    if setting.output_midi:
        if not os.path.exists(setting.output_folder):
            os.makedirs(setting.output_folder)

    for file in file_list:
        print(file.split('/')[-1])
        song = read_midi(file, model=model)
        V = integrated_stream(song, method=setting.method, texture=setting.texture)
        output_result(V, ls=metrics_ls, texture=setting.texture)

    if setting.texture == 'polyphony':
        columns = ['file', 'acc_in_note', 'acc_in_frame']
        df = pd.DataFrame(metrics_ls, columns=columns)
        df.to_csv(setting.metrics_file, index=False)

    if setting.texture == 'homophony':
        columns = ['file', 'precision_in_note', 'recall_in_note', 'f1_in_note', 
                   'precision_in_frame', 'recall_in_frame', 'f1_in_frame']
        df = pd.DataFrame(metrics_ls, columns=columns)
        df.to_csv(setting.metrics_file, index=False)


    
