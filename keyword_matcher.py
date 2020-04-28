import pandas as pd
import numpy as np

class KeywordMatcher:

    def __init__(self, clusterer, keyword_finder, default_centroid, words_per_cluster=5):
        self.default_centroid = default_centroid
        self.clusterer = clusterer
        cluster_count = self.clusterer.cluster_count
        self.df = pd.DataFrame(columns=['keyword', 'cluster_id'])
        for i in range(cluster_count):
            centroid = self.clusterer.get_cluster_centroid(i)
            keywords = keyword_finder.top_n_keywords(centroid, words_per_cluster)
            for word in keywords:
                row = pd.DataFrame(data=[[word, i]], columns=['keyword', 'cluster_id'])
                self.df = pd.concat([self.df, row])

    def get_kw_centroid(self, keywords):
        ids = []
        for keyword in keywords:
            ids += list(self.df[self.df['keyword'] == keyword]['cluster_id'])
        if len(ids) == 0:
            return self.default_centroid
        centroid = []
        for id in ids:
            centroid.append(self.clusterer.get_cluster_centroid(id))
        return np.mean(centroid, axis=0)
