from sklearn.cluster import SpectralClustering
import numpy as np


class Clusterer:
    """This class encapsulates functionality of clustering
    organizations. 
    """

    def __init__(self, org_dataset, org_vectorspace, n_clusters=10):
        self.dataset = org_dataset
        self.vs = org_vectorspace
        org_descs = self.dataset.get_org_descriptions()
        vecs = self.vs.transform(org_descs)
        labels = SpectralClustering(n_clusters=n_clusters).fit_predict(vecs)
        self.df = self.dataset.dataframe
        self.df['clusters'] = labels

    def get_cluster_centroid(self, cluster_num):
        df = self.df[self.df['clusters']==cluster_num]
        descs = df['orgPurpose'].to_numpy()
        vecs = self.vs.transform(descs)
        return np.mean(vecs, axis=0)
