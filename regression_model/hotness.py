import pandas as pd
import numpy as np
from collections import defaultdict

class Hotness:
    def __init__(self):
        self.rank_id = dict()
        self.beat_a_rank = defaultdict(lambda : [])
        self.beat_c_rank = defaultdict(lambda : [])
        self.remix_a_rank = defaultdict(lambda : [])
        self.remix_c_rank = defaultdict(lambda : [])
        self.normal_a_rank = defaultdict(lambda : [])
        self.normal_c_rank = defaultdict(lambda : [])
    
    def __preprocess(self, data):
        rank = pd.read_csv(f'metadata/{data}_rank.csv')
        for _, row in rank.iterrows():
            self.rank_id[row[0]] = row[1]
        
        info = pd.read_csv(f'metadata/{data}_info.tsv', sep='\t')
        for _, row in info.iterrows():
            if row[0] not in self.rank_id:
                continue
            song_id = row[0]
            title = row[1].lower()
            a_id_list = row[3].strip().replace('.', ',').split(',')
            c_id_list = row[5].strip().replace('.', ',').split(',')
            r = self.rank_id[song_id]
            if 'beat' in title:
                for a in a_id_list:
                    self.beat_a_rank[a].append(r)
                for c in c_id_list:
                    self.beat_c_rank[c].append(r)
            elif 'remix' in title:
                for a in a_id_list:
                    self.remix_a_rank[a].append(r)
                for c in c_id_list:
                    self.remix_c_rank[c].append(r)
            else:
                for a in a_id_list:
                    self.normal_a_rank[a].append(r)
                for c in c_id_list:
                    self.normal_c_rank[c].append(r)
    
    def __hotness(self, id_list, rank_list):
        nor_val = 6.0
        a = []
        for song_id in id_list:
            v = np.array(rank_list[song_id])
            a.append(np.sqrt(np.mean(v ** 2)) if v.size > 0 else nor_val)
        a = np.array(a + [nor_val])
        return np.sqrt(np.mean(a ** 2))
    
    def __hotness_calculate(self, data):
        a_hotness = dict()
        c_hotness = dict()
        info = pd.read_csv(f'metadata/{data}_info.tsv', sep='\t')
        for _, row in info.iterrows():
            song_id = row[0]
            title = row[1].lower()
            a_id_list = row[3].strip().replace('.', ',').split(',')
            c_id_list = row[5].strip().replace('.', ',').split(',')
            if 'beat' in title:
                a_hotness[song_id] = self.__hotness(a_id_list, self.beat_a_rank)
                c_hotness[song_id] = self.__hotness(c_id_list, self.beat_c_rank)
            elif 'remix' in title:
                a_hotness[song_id] = self.__hotness(a_id_list, self.remix_a_rank)
                c_hotness[song_id] = self.__hotness(c_id_list, self.remix_c_rank)
            else:
                a_hotness[song_id] = self.__hotness(a_id_list, self.normal_a_rank)
                c_hotness[song_id] = self.__hotness(c_id_list, self.normal_c_rank)
        f = open(f'metadata/hotness_a_{data}.txt', 'w+')
        for song_id, hotness in a_hotness.items():
            print('%d %.4f' % (song_id, hotness), file=f)
        f.close()
        f = open(f'metadata/hotness_c_{data}.txt', 'w+')
        for song_id, hotness in c_hotness.items():
            print('%d %.4f' % (song_id, hotness), file=f)
        f.close()

    def compute(self):
        self.__preprocess('train')
        self.__preprocess('test')
        self.__hotness_calculate('train')
        self.__hotness_calculate('test')

    def clear(self):
        self.rank_id.clear()
        self.beat_a_rank.clear()
        self.beat_c_rank.clear()
        self.remix_a_rank.clear()
        self.remix_c_rank.clear()
        self.normal_a_rank.clear()
        self.normal_c_rank.clear()

hotness = Hotness()
hotness.compute()
