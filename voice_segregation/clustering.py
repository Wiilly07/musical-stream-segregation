from sklearn.cluster import SpectralClustering
from voice_segregation.basic_class import Song, Voice
from tensorflow import keras
import numpy as np

def spectrum_system(song, n, ideal=False, window=40, step=10):
    voices = []
    if ideal:
        dis_m = song.label_sp_dis_m.copy()
    else:
        dis_m = song.sp_dis_m.copy() 
    
    if len(song) <= 40: 
        cluster = SpectralClustering(n_clusters=n, assign_labels="discretize", affinity='precomputed')
        cluster.fit(dis_m)
        v = Voice(song=song)
        v.read(label=cluster.labels_)
        voices.append(v)

    else:
        for i in range((len(song)-window)//step + 2):
            cluster = SpectralClustering(n_clusters=n, assign_labels="discretize", affinity='precomputed')

            if i*step+window+5 > len(song):
                cluster.fit(dis_m[i*step:, i*step:])

                v = Voice(song=song)
                v.read(start=i*step, label=cluster.labels_)
                voices.append(v)
                break

            else:
                cluster.fit(dis_m[i*step:i*step+window, i*step:i*step+window])

                v = Voice(song=song)
                v.read(start=i*step, end=i*step+window-1, label=cluster.labels_)
                voices.append(v)
            
    return voices 

def predict_k(song, window=40, step=10):
    tracks = []
    dis_m = song.sp_dis_m.copy()
    
    if len(song) <= window:
        tracks.append(predict_cluster_n(top_ev(laplacian(dis_m))))
    else:
        for i in range((len(song)-window)//step + 2):
            if i*step+window+5 > len(song):
                tracks.append(predict_cluster_n(top_ev(laplacian(dis_m[i*step:, i*step:]))))
                break
            else:
                tracks.append(predict_cluster_n(top_ev(laplacian(dis_m[i*step:i*step+window, i*step:i*step+window]))))

    return max(set(tracks), key=tracks.count)

def pitch_prox(ls):

    track_num = len(ls[0])
    V_total_dict = {f'voice_{k}':[] for k in range(track_num)}

    def cal_pitch_mean(ls_of_nn, song): # note numbers
        return song.df_multitrack.iloc[ls_of_nn].note.mean()

    for V in ls:
        V_undict = [i for i in V.voices.values()]
        V_pitch = [cal_pitch_mean(ls_of_nn, V.song) for ls_of_nn in V_undict]
        V_sorted = [x for _, x in sorted(zip(V_pitch, V_undict), key=lambda pair: pair[0])]

        for i, track in enumerate(V_sorted):
            V_total_dict[f'voice_{i}'] += track

    return Voice(song=ls[0].song, Dict=V_total_dict)


def minimal_overlap(ls, filling=True):
    if len(ls) == 1:
        return ls[0]
    else:
        while(1):
            new_ls = [a+b for a, b in zip(ls[0::2], ls[1::2])]
            if filling:
                for v in new_ls:
                    v.filling_missing_notes()
            if len(ls) == 2:
                return new_ls[0]

            if len(ls) % 2 == 1:
                new_ls.append(ls[-1])

            ls = new_ls

def laplacian(M):
    W = M.copy()
    np.fill_diagonal(W, 0)
    Diag = np.sum(W, axis=1)
    D = np.diag(Diag)
    return D - W

def top_ev(L, top=10):
    ev = np.sort(np.linalg.eigvals(L))[:top]
    return ev

def predict_cluster_n(ev):
    return np.argmax(np.diff(ev))+1
