# -*- encoding: utf-8 -*-
"""
Clustering of distance matrixes.

:class:`Clustering` represents a hierarchical clustering which can be flattened
using :meth:`Clustering.fcluster`, the flattened clustering is then represented
by :class:`FlatClustering`.

If supported by the installed version of scikit-learn, there is also a
KMedoidsClustering.
"""

import logging
logger = logging.getLogger(__name__)

from pprint import pformat
import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from delta.util import Metadata
from delta.deltas import DistanceMatrix
from delta.corpus import Corpus
from sklearn import metrics


class Clustering:
    """
    Represents a hierarchical clustering.

    Note:
        This is subject to refactoring once we implement more clustering
        methods
    """

    def __init__(self, distance_matrix, method="ward", **kwargs):
        self.metadata = Metadata(distance_matrix.metadata,
                                 cluster_method=method, **kwargs)
        self.distance_matrix = distance_matrix
        self.describer = distance_matrix.document_describer
        self.method = method
        self.linkage = self._calc_linkage()

    def _calc_linkage(self):
        if self.method == "ward":
            return sch.linkage(self.distance_matrix, method="ward",
                               metric="euclidean")
        else:
            return sch.linkage(ssd.squareform(self.distance_matrix),
                               method=self.method, metric="euclidean")

    def fclustering(self):
        """
        Returns a default flat clustering from the hierarchical version.

        This method uses the :class:`DocumentDescriber` to determine the
        groups, and uses the number of groups as a maxclust criterion.

        Returns:
            FlatClustering: A properly initialized representation of the flat
            clustering.
        """
        flat = FlatClustering(self.distance_matrix, metadata=self.metadata,
                              flattening='maxclust')
        flat.set_clusters(sch.fcluster(self.linkage, flat.group_count,
                                       criterion="maxclust"))
        return flat


class FlatClustering:
    """
    A flat clustering represents a non-hierarchical clustering.

    Notes:
        FlatClustering uses a data frame field called ``data`` to store the
        actual clustering.  This field will have the same index as the distance
        matrix, and three columns labeled ``Group``, ``GroupID``, and
        ``Cluster``.  ``Group`` will be the group label returned by the
        :class:`DocumentDescriber` we use, ``GroupID`` a numerical ID for each
        group (to be used as ground truth) and ``Cluster`` the numerical ID of
        the actual cluster associated by the clustering algorithm.

        As long as FlatClusterings ``initialized`` property is ``False``, the
        Clustering is not assigned yet.

    """

    def __init__(self, distances, clusters=None, metadata=None, **kwargs):
        if isinstance(distances, DistanceMatrix):
            self.distances = distances
            self.document_describer = self.distances.document_describer
            self.documents = self.distances.index
        elif isinstance(distances, Corpus):
            self.distances = None
            self.document_describer = distances.document_describer
            self.documents = distances.index
        else:
            raise ValueError(
                "Flat clustering must be initialized from a distance matrix or a corpus.\n"
                "(did you want to call Clustering.fcluster() instead?)")
        self.distances = distances
        self.metadata = Metadata(metadata if metadata is not None else
                                 distances.metadata, **kwargs)
        self.data, self.group_count = self._init_data()
        self.initialized = False
        if clusters is not None:
            self.set_clusters(clusters)

    def set_clusters(self, clusters):
        if self.initialized:
            raise Exception("Already initialized")
        self.data["Cluster"] = clusters
        self.initialized = True

    def _init_data(self):
        clustering = pd.DataFrame(index=self.distances.index)
        dd = self.distances.document_describer
        clustering["Group"] = [dd.group_name(doc) for doc in clustering.index]
        group_count = len(dd.groups(clustering.index))
        group_idx = pd.Series(index=clustering.Group.value_counts().index,
                              data=range(0, group_count))
        clustering["GroupID"] = clustering.Group.map(group_idx)
        return clustering, group_count

    @staticmethod
    def ngroups(df):
        """
        With df being a data frame that has a Group column, return the number
        of different authors in df.
        """
        return len(set(df.Group))

    def cluster_errors(self):
        """
        Calculates the number of cluster errors by:

        1. calculating the total number of different authors in the set
        2. calling sch.fcluster to generate at most that many flat clusters
        3. for each of those clusters, the cluster errors are the number of
           authors in this cluster - 1
        4. sum of each cluster's errors = result
        """
        return int((self.data.groupby("Cluster")
                    .agg(self.ngroups).Group-1).sum())

    def purity(self):
        """
        To compute purity, each cluster is assigned to the class which is most
        frequent in the cluster, and then the accuracy of this assignment is
        measured by counting the number of correctly assigned documents and
        dividing by $N$
        """
        def correctly_classified(cluster):
            return cluster.Group.value_counts()[0]
        return int(self.data.groupby("Cluster")
                   .agg(correctly_classified)
                   .Group.sum()) / self.data.index.size

    def entropy(self):
        """
        Smaller entropy values suggest a better clustering.
        """
        classes = self.data.Group.unique().size

        def cluster_entropy(cluster):
            class_counts = cluster.value_counts()
            return float((class_counts / cluster.index.size *
                          np.log(class_counts / cluster.index.size)).sum() *
                         (-1)/np.log(classes))

        def weighted_cluster_entropy(cluster):
            return (cluster.index.size / self.data.index.size) * \
                cluster_entropy(cluster)

        return self.data.groupby("Cluster") \
            .agg(weighted_cluster_entropy).Group.sum()

    def adjusted_rand_index(self):
        """
        Calculates the Adjusted Rand Index for the given flat clustering
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score
        """
        logger.debug("Calculating ARI for %s", self.data)
        return metrics.adjusted_rand_score(self.data.GroupID,
                                           self.data.Cluster)

    def homogeneity_completeness_v_measure(self):
        return metrics.homogeneity_completeness_v_measure(self.data.GroupID,
                                                          self.data.Cluster)

    def evaluate(self):
        """
        Returns:
            pandas.Series: All scores for the current clustering
        """
        result = pd.Series()
        result["Cluster Errors"] = self.cluster_errors()
        result["Adjusted Rand Index"] = self.adjusted_rand_index()
        result["Homogeneity"], result["Completeness"], result["V Measure"] = \
            self.homogeneity_completeness_v_measure()
        result["Purity"] = self.purity()
        result["Entropy"] = self.entropy()
        return result

    def clusters(self, labeled=False):
        """
        Documents by cluster.

        Args:
            labeled (bool): If ``True``, represent each document by its *label*
                as calculated by the :class:`DocumentDescriber`. This is
                typically a human-readable, shortened description
        Returns:
            dict: Maps each cluster number to a list of documents.
        """
        clusters = self.data.groupby("Cluster").groups
        if labeled:
            dd = self.distances.document_describer
            return {n: [dd.label(doc) for doc in docs]
                    for n, docs in clusters.items()}
        else:
            return clusters

    def describe(self):
        """
        Returns a description of the current flat clustering.
        """
        clusters = self.clusters(labeled=True)
        result = "{} clusters of {} documents (ground truth: {} groups):\n" \
            .format(len(clusters), len(self.data.index), self.group_count)
        result += pformat(clusters, compact=True) + '\n'
        return result

try:
    from sklearn.cluster import KMedoids

    class KMedoidsClustering(FlatClustering):

        def __init__(self, distances, n_clusters=None, metadata=None,
                     **kwargs):
            super().__init__(distances, metadata, **kwargs)
            if n_clusters is None:
                n_clusters = self.group_count
            model = KMedoids(n_clusters=n_clusters, **kwargs)
            self.set_clusters(model.fit_predict(distances))

except ImportError:
    logger.log(logging.ERROR, "KMedoids clustering not available.\n" \
               "You need a patched scikit-learn, see README.txt", exc_info=1)
