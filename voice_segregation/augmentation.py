import numpy as np
from scipy.stats import truncnorm 
from itertools import combinations
from copy import deepcopy

def reorder(song):
    song.df_multitrack.sort_values(by=['start', 'note', 'duration'], ascending=[True, True, False], inplace=True)
    song.df_multitrack.reset_index(drop=True, inplace=True)

def onset_shift(song, note, length_ratio):
    offset = song.df_multitrack.loc[note, 'start'] + song.df_multitrack.loc[note, 'duration']
    new_onset = song.df_multitrack.loc[note, 'start'] + song.df_multitrack.loc[note, 'duration'] * length_ratio
    song.df_multitrack.loc[note, 'start'] = max(0, new_onset)
    song.df_multitrack.loc[note, 'duration'] = offset - song.df_multitrack.loc[note, 'start']

def offset_shift(song, note, length_ratio):
    song.df_multitrack.loc[note, 'duration'] += song.df_multitrack.loc[note, 'duration'] * length_ratio

def onset_shift_random(song):
    for i in range(len(song)):
        random_len = truncnorm.rvs(a=-1, b=1, loc=0, scale=0.15, size=1)[0]
        onset_shift(song, i, length_ratio=random_len)

def offset_shift_random(song):
    for i in range(len(song)):
        random_len = truncnorm.rvs(a=-1, b=1, loc=0, scale=0.15, size=1)[0]
        offset_shift(song, i, length_ratio=random_len)

def onoff_shift_random(song):
    if np.random.choice(2):
        onset_shift_random(song)
        offset_shift_random(song)
        clean_samepitch_overlap(song)

def clean_samepitch_overlap(song):
    for i in set(song.df_multitrack.label):
        copy_df = song.df_multitrack[song.df_multitrack.label == i].copy()
        copy_df['end'] = copy_df.start + copy_df.duration
        
        id_ls = copy_df[copy_df.note.diff() == 0].index  #candidates of pairs of samepitch
        dict_convert = {k: item for item, k in enumerate(copy_df.index)}

        for true_index in id_ls:
            relative_index = dict_convert[true_index]
            if copy_df.end.iloc[relative_index-1] > copy_df.start.iloc[relative_index]:

                copy_df.start.iloc[relative_index] = copy_df.end.iloc[relative_index-1]
                copy_df.duration[true_index] = copy_df.end[true_index] - copy_df.start[true_index]
                
                song.df_multitrack.start[true_index] = copy_df.start[true_index]
                song.df_multitrack.duration[true_index] = copy_df.duration[true_index]

    reorder(song)
   
def drop_voice(song, number):
    df_drop = song.df_multitrack.loc[song.df_multitrack.label != number].reset_index(drop=True)
    song.df_multitrack = df_drop

def drop_voice_random(song):
    drop_voice_num = np.random.choice([0, 1, 2], size=1, p=[0.6, 0.3, 0.1])[0]
    voice_cand = list(combinations([0, 1, 2, 3], drop_voice_num))
    drop_voice_index = voice_cand[np.random.choice(len(voice_cand), size=1)[0]]
    for index in drop_voice_index:
        drop_voice(song, index)
    
def speed_change(song, rate):
    song.df_multitrack.start = song.df_multitrack.start/rate
    song.df_multitrack.duration = song.df_multitrack.duration/rate

def speed_change_random(song):
    rate = 2 ** np.random.uniform(-1, 1)
    speed_change(song, rate)
    
def key_change(song, key):
    song.df_multitrack.note = song.df_multitrack.note + key

def key_change_random(song):
    key = np.random.choice(range(-6, 6), size=1)[0]
    key_change(song, key)

def drop_note(song, note):
    song.df_multitrack.drop(note, inplace=True)
    song.df_multitrack.reset_index(inplace=True, drop=True)

def drop_note_random(song):
    drop_rate = np.random.choice([0.0, 0.05, 0.1], size=1)[0]
    drop_note_index = np.random.choice(song.df_multitrack.index, size=int(drop_rate*len(song.df_multitrack)), replace=False)
    drop_note(song, drop_note_index)

def octave_shift(song, voice, octave):
    song.df_multitrack.loc[song.df_multitrack.label == voice, 'note'] += octave*12
    reorder(song)

def drop_twothird_voice(song, number, mode=0):
    voice_index = list(song.df_multitrack.loc[song.df_multitrack.label == number].index)
    voice_len = len(voice_index)
    if mode == 0:
        drop_index = voice_index[-voice_len//3*2:]  
        
    elif mode == 1:
        drop_index = voice_index[voice_len//3:-voice_len//3]
    
    else:
        drop_index = voice_index[:-len(voice_index)//3]
        
    drop_note(song, drop_index)

def drop_twothird_voice_random(song):
    current_voice_index = list(set(song.df_multitrack.label))
    drop_or_not = np.random.choice([True, False], size=1, p=[0.15, 0.85])[0]
    if drop_or_not:
        drop_voice_index = np.random.choice(current_voice_index, size=1)[0]
        mode = np.random.choice([0, 1, 2], size=1)[0]
        drop_twothird_voice(song, drop_voice_index, mode=mode)

class Augmenter:
    def __init__(self, texture, inplace=True):

        self.inplace = inplace

        if texture == 'polyphony':
            self.aug_ls = [
                drop_voice_random,
                drop_twothird_voice_random,
                drop_note_random,
                onoff_shift_random,
                key_change_random,
                speed_change_random,
            ]
        elif texture == 'homophony':
            self.aug_ls = [
                drop_note_random,
                key_change_random,
                speed_change_random,
            ]

    def __call__(self, song):
        if self.inplace:
            for aug in self.aug_ls:
                aug(song)

            reorder(song)

        else:
            song_copy = deepcopy(song)
            for aug in self.aug_ls:
                aug(song_copy)

            reorder(song_copy)
            return song_copy