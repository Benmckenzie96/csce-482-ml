from sklearn.cluster import SpectralClustering
import numpy as np


class Clusterer:
    """This class encapsulates functionality of clustering
    organizations. The clustering algorithm this class uses
    is Spectral Clustering.

    Attributes:
        cluster_count (int): The number of clusters generated.
            This will be equal to the value passed for 'n_clusters'
            in __init__.
        _dataset (OrgDataset): an OrgDataset instance holding
            the organizations you wish to cluster.
        _vs (VectorSpace): a VectorSpace instance. This instance
            must be trained before being supplied to a Clusterer
            instance.
    """

    def __init__(self, org_dataset, org_vectorspace, n_clusters=10):
        """Initializes instance. Note that clustering is performed
        in this function.
        """
        self._dataset = org_dataset
        self._vs = org_vectorspace
        self.cluster_count = n_clusters
        org_descs = self._dataset.get_org_descriptions()
        vecs = self._vs.transform(org_descs)
        labels = SpectralClustering(n_clusters=n_clusters).fit_predict(vecs)
        self.df = self._dataset.dataframe
        self.df['clusters'] = labels

    def get_cluster_centroid(self, cluster_num):
        """Gets the centroid of the cluster with the
        specified label.

        Args:
            cluster_num (int): The label of the cluster
                you wish to fetch the centroid of.

        Returns:
            a numpy array containing the cluster centroid.
        """
        df = self.df[self.df['clusters']==cluster_num]
        descs = df['orgPurpose'].to_numpy()
        vecs = self._vs.transform(descs)
        return np.mean(vecs, axis=0)
