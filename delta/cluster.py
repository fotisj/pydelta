# -*- encoding: utf-8 -*-
"""
"""

import logging
logger = logging.getLogger(__name__)

from pprint import pformat
import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
import matplotlib as mpl
import matplotlib.pyplot as plt
# from scipy import linalg
# from scipy.misc import comb
# from itertools import combinations
# from functools import update_wrapper
from .util import Metadata
# from .corpus import Corpus
from sklearn import metrics


class Clustering:
    """
    Represents a clustering.

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


class Dendrogram:
    """
    Creates a dendrogram representation from a hierarchical clustering.

    This is a wrapper around, and an improvement to, :func:`sch.dendrogram`,
    tailored for the use in pydelta.

    Args:
        clustering (Clustering): A hierarchical clustering.
        describer (DocumentDescriber): Document describer used for determining
            the groups and the labels for the documents used (optional). By
            default, the document describer inherited from the clustering is
            used.
        ax (mpl.axes.Axes): Axes object to draw on. Uses pyplot default axes if
            not provided.
        orientation (str): Orientation of the dendrogram. Currently, only
            "right" is supported (default).
        font_size: Font size for the label, in points. If not provided,
            :func:`sch.dendrogram` calculates a default.
        link_color (str): The color used for the links in the dendrogram, by
            default ``k`` (for black).
        title (str): a title that will be printed on the plot. The string may
           be a template string as supported by :meth:`str.format_map` with
           metadata field names in curly braces, it will be evaluated against
           the clustering's metadata. If you pass ``None`` here, no title will
           be added.

    Notes:
        The dendrogram will be painted by matplotlib / pyplot using the default
        styles, which means you can use, e.g., :module:`seaborn` to influence
        the overall design of the image.

        :class:`Dendrogram` handles coloring differently than
        :func:`sch.dendrogram`: It will color the document labels according to
        the pre-assigned grouping (e.g., by author). To do so, it will build on
        matplotlib's default color_cycle, and it will rotate, so if you need
        more colors, adjust the color_cycle accordingly.
    """

    def __init__(self, clustering, describer=None, ax=None,
                 orientation="left", font_size=None, link_color="k",
                 title="Corpus: {corpus}",
                 xlabel="Delta: {delta}, {words} most frequent {features}"):

        self.clustering = clustering
        self.linkage = clustering.linkage
        self.metadata = clustering.metadata
        self.describer = clustering.describer \
            if describer is None else describer
        self.documents = list(clustering.distance_matrix.index)
        self.orientation = orientation
        self._init_colormap()

        plt.clf()
        self.dendro_data = sch.dendrogram(self.linkage,
                                          orientation=orientation,
                                          labels=self.documents,
                                          leaf_rotation = 0 if orientation == 'left' else 90,
                                          ax=ax,
                                          link_color_func=lambda k: link_color)

        # Now redo the author labels. To do so, we map a color to each author
        # (using the describer) and then
        self.ax = plt.gca() if ax is None else ax
        self.fig = plt.gcf()
        self._relabel_axis()
        if title is not None:
            plt.title(title.format_map(self.metadata))
        if xlabel is not None:
            plt.xlabel(xlabel.format_map(self.metadata))
        plt.tight_layout(2)

    def link_color_func(self, k):
        print(k)
        return "k"

    def _init_colormap(self):
        groups = self.describer.groups(self.documents)
        colors = mpl.rc_params()['axes.color_cycle']
        self.colormap = {group: colors[idx % len(colors)]
                         for idx, group in enumerate(groups)}
        self.colorlist = [self.colormap[self.describer.group_name(doc)]
                          for doc in self.documents]
        return self.colormap

    def _relabel_axis(self):
        if self.orientation == 'left':
            labels = self.ax.get_ymajorticklabels()
        else:
            labels = self.ax.get_xmajorticklabels()
        display_labels = []
        for label in labels:
            group = self.describer.group_name(label.get_text())
            label.set_color(self.colormap[group])
            display_label = self.describer.label(label.get_text())
            label.set_text(display_label)       # doesn't really set the labels
            display_labels.append(display_label)
        if self.orientation == 'left':
            self.ax.set_yticklabels(display_labels)
        else:
            self.ax.set_xticklabels(display_labels)


    def show(self):
        plt.show()

    def save(self, fname, **kwargs):
        self.fig.savefig(fname, **kwargs)







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
        self.distances = distances
        self.metadata = Metadata(metadata if metadata is not None else
                                 distances.metadata, **kwargs)
        self.data, self.group_count = self._init_data()
        if clusters is None:
            self.initialized = False
        else:
            self.data["Clustering"] = clusters
            self.initialized = True

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
