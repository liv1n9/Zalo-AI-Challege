import zipfile
import pandas as pd
import librosa
import os
import numpy as np
import multiprocessing

def process(song_id, data):
    try:
        y, sr = librosa.load(f'data/{data}/{song_id}.mp3', mono=True, duration=240, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=39)
        np.savetxt(f'/media/livw/Data/temp_data/{data}/mfcc_{song_id}.txt', mfcc, delimiter=' ', fmt='%1.2f')
        print(f'Done {song_id}')
    except:
        print(f'Song {song_id} got error')

def extract_feature(data):
    info = pd.read_csv(f'metadata/{data}_info.tsv', sep='\t')
    empty_song = set([1075810992, 1076340538, 1076340539, 1076340540, 1076340541, 1076340542, 1076340543, 1076340544])
    process_list = []

    for index, row in info.iterrows():
        print(index)
        song_id = row[0]
        if song_id not in empty_song:
            t = multiprocessing.Process(target=process, args=(song_id, data,))
            t.start()
            process_list.append(t)
        if len(process_list) == 4:
            for p in process_list:
                p.join()
            process_list.clear()
    for p in process_list:
        p.join()
