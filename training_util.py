import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from voice_segregation.basic_class import Song
import voice_segregation.augmentation as aug
import setting
import concurrent.futures
import numpy as np
from functools import partial
from tensorflow import keras
from sklearn.utils import shuffle
from contextlib import redirect_stdout

import warnings
warnings.filterwarnings("ignore")

def song_aug(filename, augment=True):
    sample_song = Song(filename, reading_range=setting.reading_range)
    if setting.texture == 'polyphony':
        sample_song.delete_same_note()

    elif setting.texture == 'homophony':
        sample_song.delete_accompaniment()

    if augment:
        augmenter = aug.Augmenter(texture=setting.texture)
        augmenter(sample_song)

    X, y = sample_song.pairing_notes(max_dist=setting.max_dist)
    return X, y

def data_pipe(file_list, augment=True):
    func = partial(song_aug, augment=augment)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(func, file_list)
    results = list(results)
    X_list, y_list = zip(*results)

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(filters=32, kernel_size=32, activation='relu', input_shape=(setting.reading_range*2+1, 6)))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=16, activation='relu'))
    model.add(keras.layers.Conv1D(filters=64, kernel_size=16, activation='relu'))
    model.add(keras.layers.Conv1D(filters=64, kernel_size=8, activation='relu'))
    model.add(keras.layers.Conv1D(filters=128, kernel_size=8, activation='relu'))
    model.add(keras.layers.Conv1D(filters=128, kernel_size=4, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=2, activation='softmax'))
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_one_epoch(model, augment, file_list, val_data, batch_size):
    X, y = data_pipe(file_list=file_list, augment=augment)
    model.fit(X, y, batch_size=batch_size, 
              epochs=1, verbose=2, validation_data=val_data)
        
def training(file_list, val_file_list, epochs, batch_size, aug, model_file=None):
    
    try:
        model = keras.models.load_model(model_file)

    except:
        print('no model, building one')
        model = build_model()
        aug = False

        print('first training...')

    file_list = shuffle(file_list)
    pile = setting.pile
    t = int(np.ceil(len(file_list)/pile))

    for j in range(epochs):
        for i in range(t-1):
            hist = train_one_epoch(model=model, augment=aug, file_list=file_list[i*pile:(i+1)*pile], val_data=None, batch_size=batch_size)
        X_val, y_val = data_pipe(file_list=val_file_list, augment=False)
        #hist = train_one_epoch(model=model, augment=aug, file_list=file_list[(i+1)*pile:], val_data=(X_val, y_val), batch_size=batch_size)
        loss, acc = model.evaluate(X_val, y_val)
        
    return model, acc


def training_util():    
    train_list = [os.path.join(setting.train_folder, i) for i in os.listdir(setting.train_folder)]
    val_list = [os.path.join(setting.val_folder, i) for i in os.listdir(setting.val_folder)]

    model_trained, valid_acc = training(file_list=train_list, val_file_list=val_list, 
                                        epochs=1, batch_size=setting.batch_size, 
                                        aug=True,  model_file=setting.model_file)
    return model_trained, valid_acc


if __name__ == '__main__':
    best_acc = 0
    with open(setting.log_file, 'a') as fp:
        for i in range(setting.epoch):
            print(f'epoch: {i+1}', file=fp)
            model_trained, valid_acc = training_util()
            if valid_acc >= best_acc:
                model_trained.save(setting.model_file)
                best_acc = valid_acc
            print(f'valid acc: {valid_acc}, best acc: {best_acc}\n', file=fp)
