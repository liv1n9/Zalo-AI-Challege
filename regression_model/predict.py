from sklearn.linear_model import ElasticNet, LinearRegression
import pandas as pd
import numpy as np
from regression_model.hotness import Hotness

def predict():
    # Train model
    rank_id = dict()
    train_rank = pd.read_csv('metadata/train_rank.csv')
    for _, row in train_rank.iterrows():
        rank_id[row[0]] = row[1]

    beat_id = set()
    remix_id = set()
    train_info = pd.read_csv('metadata/train_info.tsv', sep='\t')
    for _, row in train_info.iterrows():
        song_id = row[0]
        title = row[1].lower()
        if 'beat' in title:
            beat_id.add(song_id)
        elif 'remix' in title:
            remix_id.add(song_id)

    X_beat_train = []
    y_beat_train = []
    X_remix_train = []
    y_remix_train = []
    X_normal_train = []
    y_normal_train = []
    train_feature = pd.read_csv('feature_data/feature_train.txt', sep=' ', header=None)
    for _, row in train_feature.iterrows():
        song_id = row[0]
        feature = row[1:]
        r = float(rank_id[song_id])
        if song_id in beat_id:
            X_beat_train.append(feature)
            y_beat_train.append(r)
        elif song_id in remix_id:
            X_remix_train.append(feature)
            y_remix_train.append(r)
        else:
            X_normal_train.append(feature)
            y_normal_train.append(r)
    
    print('Training...')
    regr_beat = ElasticNet()
    regr_remix = ElasticNet()
    regr_normal = LinearRegression()
    regr_beat.fit(np.array(X_beat_train), np.array(y_beat_train))
    regr_remix.fit(np.array(X_remix_train), np.array(y_remix_train))
    regr_normal.fit(np.array(X_normal_train), np.array(y_normal_train))

    # Predict
    beat_id.clear()
    remix_id.clear()
    test_info = pd.read_csv('metadata/test_info.tsv', sep='\t')
    for _, row in test_info.iterrows():
        song_id = row[0]
        title = row[1].lower()
        if 'beat' in title:
            beat_id.add(song_id)
        elif 'remix' in title:
            remix_id.add(song_id)

    test_feature = pd.read_csv('feature_data/feature_test.txt', sep=' ', header=None)
    f = open('metadata/test_rank.csv', 'w+')
    print('ID,label', file=f)
    g = open('my_submission.csv', 'w+')
    print('Predicting...')
    for _, row in test_feature.iterrows():
        song_id = row[0]
        feature = row[1:]
        if song_id in beat_id:
            p = regr_beat.predict([feature])
        elif remix_id in beat_id:
            p = regr_remix.predict([feature])
        else:
            p = regr_normal.predict([feature])
        if p[0] < 1:
            p[0] = 1
        if p[0] > 10:
            p[0] = 10
        print('%d,%.4f' % (song_id, p[0]), file=f)
        print('%d,%.4f' % (song_id, p[0]), file=g)
    f.close()
    g.close()

def semi_supervised():
    epoch = 1
    for i in range(epoch):
        hotness = Hotness()
        hotness.compute()
        predict()

if __name__ == '__main__':
    semi_supervised()
