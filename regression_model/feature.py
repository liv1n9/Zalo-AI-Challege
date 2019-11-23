import pandas as pd
import numpy as np
from collections import defaultdict

def create_feature(data):
    info = pd.read_csv(f'metadata/{data}_info.tsv', sep='\t')
    f = open(f'feature_data/feature_{data}.txt', 'w+')

    features = defaultdict(lambda: [])
    for index, row in info.iterrows():
        try:
            print(index)
            g = open(f'/media/livw/Data/temp_data/{data}/mfcc_{row[0]}.txt', 'r')
            for line in g.readlines():
                a = np.array(list(map(float, line.split())))
                features[int(row[0])].append(np.cbrt(np.mean(a ** 3)))
                features[int(row[0])].append(a.mean())
                features[int(row[0])].append(np.std(a))
            g.close()
        except FileNotFoundError:
            pass

    a_hotness = pd.read_csv(f'metadata/hotness_a_{data}.txt', sep=' ', header=None)
    for index, row in a_hotness.iterrows():
        song_id = int(row[0])
        if song_id in features:
            features[song_id].append(row[1])

    c_hotness = pd.read_csv(f'metadata/hotness_c_{data}.txt', sep=' ', header=None)
    for index, row in c_hotness.iterrows():
        song_id = int(row[0])
        if song_id in features:
            features[song_id].append(row[1])

    for song_id, feature in features.items():
        print(song_id, end='', file=f)
        for x in feature:
            print(' %.4f' % x, end='', file=f)
        print(file=f)
    f.close()
