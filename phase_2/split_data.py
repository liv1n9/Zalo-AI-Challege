import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression
from collections import defaultdict
from sklearn.model_selection import cross_val_score
import time

rank_id = dict()

train_rank = pd.read_csv('metadata/train_rank.csv')
for index, row in train_rank.iterrows():
    rank_id[row[0]] = row[1]

beat_id = set()
remix_id = set()

beat_artist_rank = defaultdict(lambda : [])
beat_composer_rank = defaultdict(lambda : [])
remix_artist_rank = defaultdict(lambda : [])
remix_composer_rank = defaultdict(lambda : [])
normal_artist_rank = defaultdict(lambda : [])
normal_composer_rank = defaultdict(lambda : [])

a_hotness = dict()
c_hotness = dict()
train_timestamp = dict()
nor_val = {
    'beat': 6.0,
    'remix': 6.0,
    'normal': 5.0
}

def hotness(l, d, t):
    a = []
    v = nor_val[t]
    for i in l:
        x = np.array(d[i])
        a.append(np.sqrt(np.mean(x ** 2)) if x.size > 0 else v)
    a = np.array(a + [v])
    return np.sqrt(np.mean(a ** 2))

train_info = pd.read_csv('metadata/train_info.tsv', sep='\t')
for index, row in train_info.iterrows():
    title = row[1].strip().lower()
    a_id_list = row[3].strip().replace('.', ',').split(',')
    c_id_list = row[5].strip().replace('.', ',').split(',')
    train_timestamp[row[0]] = time.mktime(time.strptime(row[6], "%Y-%m-%d %H:%M:%S"))
    if 'beat' in title:
        beat_id.add(row[0])
        for a in a_id_list:
            beat_artist_rank[a].append(rank_id[row[0]])
        for c in c_id_list:
            beat_composer_rank[c].append(rank_id[row[0]])
    elif 'remix' in title:
        remix_id.add(row[0])
        for a in a_id_list:
            remix_artist_rank[a].append(rank_id[row[0]])
        for c in c_id_list:
            remix_composer_rank[c].append(rank_id[row[0]])
    else:
        for a in a_id_list:
            normal_artist_rank[a].append(rank_id[row[0]])
        for c in c_id_list:
            normal_composer_rank[c].append(rank_id[row[0]])
    

for index, row in train_info.iterrows():
    a_id_list = row[3].strip().replace('.', ',').split(',')
    c_id_list = row[5].strip().replace('.', ',').split(',')
    if row[0] in beat_id:
        a_hotness[row[0]] = hotness(a_id_list, beat_artist_rank, 'beat')
        c_hotness[row[0]] = hotness(c_id_list, beat_composer_rank, 'beat')
    elif row[0] in remix_id:
        a_hotness[row[0]] = hotness(a_id_list, remix_artist_rank, 'remix')
        c_hotness[row[0]] = hotness(c_id_list, remix_composer_rank, 'remix')
    else:
        a_hotness[row[0]] = hotness(a_id_list, normal_artist_rank, 'normal')
        c_hotness[row[0]] = hotness(c_id_list, normal_composer_rank, 'normal')

X_normal = defaultdict(lambda: [])
y_normal = []

X_beat = defaultdict(lambda: [])
y_beat = []

X_remix = defaultdict(lambda: [])
y_remix = []

train_mfcc = pd.read_csv('train_data/train_mfcc.txt', sep=' ')
for index, row in train_mfcc.iterrows():
    song_id = row[0]
    data = row[1:].tolist() + [a_hotness[song_id]] + [c_hotness[song_id]] + [train_timestamp[song_id]]
    if song_id in beat_id:
        X_beat[song_id] += data
        y_beat.append(rank_id[song_id])
    elif song_id in remix_id:
        X_remix[song_id] += data
        y_remix.append(rank_id[song_id])
    else:
        X_normal[song_id] += data
        y_normal.append(rank_id[song_id])

train_chroma = pd.read_csv('train_data/train_chroma_stft.txt', sep=' ', header=None)
for index, row in train_chroma.iterrows():
    song_id = row[0]
    data = row[1:].tolist()
    if song_id in beat_id:
        X_beat[song_id] += data
    elif song_id in remix_id:
        X_remix[song_id] += data
    else:
        X_normal[song_id] += data

regr_beat = ElasticNet()
regr_remix = ElasticNet()
regr_normal = LinearRegression()

rmse_beat = np.sqrt(cross_val_score(regr_beat, list(X_beat.values()), y_beat, cv=10, scoring='neg_mean_squared_error') * -1).mean()
rmse_remix = np.sqrt(cross_val_score(regr_remix, list(X_remix.values()), y_remix, cv=10, scoring='neg_mean_squared_error') * -1).mean()
rmse_normal = np.sqrt(cross_val_score(regr_normal, list(X_normal.values()), y_normal, cv=10, scoring='neg_mean_squared_error') * -1).mean()
print(rmse_beat, rmse_remix, rmse_normal)

regr_beat.fit(np.array(list(X_beat.values())), np.array(y_beat))
regr_remix.fit(np.array(list(X_remix.values())), np.array(y_remix))
regr_normal.fit(np.array(list(X_normal.values())), np.array(y_normal))

beat_test_id = set()
remix_test_id = set()

test_timestamp = dict()
a_hotness_test = dict()
c_hotness_test = dict()
test_info = pd.read_csv('metadata/test_info.tsv', sep='\t')
for index, row in test_info.iterrows():
    title = row[1].strip().lower()
    a_id_list = row[3].strip().replace('.', ',').split(',')
    c_id_list = row[5].strip().replace('.', ',').split(',')
    test_timestamp[row[0]] = time.mktime(time.strptime(row[6], "%Y-%m-%d %H:%M:%S"))
    if 'beat' in title:
        beat_test_id.add(row[0])
        a_hotness_test[row[0]] = hotness(a_id_list, beat_artist_rank, 'beat')
        c_hotness_test[row[0]] = hotness(c_id_list, beat_composer_rank, 'beat')
    elif 'remix' in title:
        remix_test_id.add(row[0])
        a_hotness_test[row[0]] = hotness(a_id_list, remix_artist_rank, 'remix')
        c_hotness_test[row[0]] = hotness(c_id_list, remix_composer_rank, 'remix')
    else:
        a_hotness_test[row[0]] = hotness(a_id_list, normal_artist_rank, 'normal')
        c_hotness_test[row[0]] = hotness(c_id_list, normal_composer_rank, 'normal')
    
    

test_mfcc = pd.read_csv('train_data/test_mfcc.txt', sep=' ')
test_chroma = pd.read_csv('train_data/test_chroma_stft.txt', sep=' ', header=None)
g = open('regr_submission.csv', 'w+')
temp = test_mfcc.values
for i in range(temp.shape[0]):
    data = [temp[i][1:].tolist() + [a_hotness_test[temp[i][0]]] + [c_hotness_test[temp[i][0]]] + [test_timestamp[temp[i][0]]] + test_chroma.values[i][1:].tolist()]
    if temp[i, 0] in beat_test_id:
        p = regr_beat.predict(data)
    elif temp[i, 0] in remix_test_id:
        p = regr_remix.predict(data)
    else:
        p = regr_normal.predict(data)
    if p[0] < 1:
        p[0] = 1
    if p[0] > 10:
        p[0] = 10
    print('%d,%.4f' % (temp[i, 0], p[0]), file=g)