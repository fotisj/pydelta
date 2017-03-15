# -*- encoding: utf-8 -*-
"""
Various visualization tools.
"""

import logging
logger = logging.getLogger(__name__)

import scipy.cluster.hierarchy as sch
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import manifold, decomposition
from sklearn.base import TransformerMixin
from collections.abc import Iterable


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
                 xlabel="Delta: {delta_title}, {words} most frequent {features}"):

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
        props = mpl.rcParams['axes.prop_cycle']
        self.colormap = {x: y['color'] for x,y in zip(groups, props())}
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


def scatterplot_delta(deltas,
                      red_f=manifold.MDS(dissimilarity="precomputed", n_jobs=-1)):
    """
    deltas: pydelta dist. matrix
    red_f: func for dimensionality reduction, e.g. "decomposition.PCA(n_components=2)"

    return: plot?
    """
    if red_f == "mds":
        red_f = manifold.MDS(dissimilarity="precomputed", n_jobs=-1)
    elif red_f == "pca":
        red_f = decomposition.PCA(n_components=2)
    elif not isinstance(red_f, TransformerMixin):
        raise ValueError('red_f must be "mds", "pca", or a Transformer, but is '
                         + repr(red_f))

    X_red = red_f.fit_transform(deltas)
    group_map = {y:x for x,y in enumerate(deltas.document_describer.groups(deltas.index))}
    label_names = [ deltas.document_describer.group_label(x) for x in deltas.index ]
    cluster_labels = [ float(group_map[deltas.document_describer.group_name(x)])/len(group_map) for x in deltas.index ]
    colors = mpl.spectral(cluster_labels)

    plt.scatter(X_red[:, 0], X_red[:, 1], marker='o', s=30, lw=0, alpha=0.7, c=colors)

    for label, color in dict(zip(label_names, colors)).items():
        plt.scatter([], [], marker='o', s=30, lw=0, alpha=0.7, c=color, label=label)
    plt.legend()
    return plt.gca()

def _prep_slice(arg):
    """
    Prepare an argument to be passed to an indexer's __getitem__.

    Arg can be none (``:``), an list (passed through), a slice (passed
    through), or sth else like an integer (``:n``)
    """
    if arg is None:
        return slice(None)
    elif isinstance(arg, Iterable) or isinstance(arg, slice):
        return arg
    else:
        return slice(arg)

def spikeplot(corpus, docs=slice(None), features=50, figsize=None, **kwargs):
    """
    Prepares a spike plot of a (normalized) corpus.

    Args:
        corpus (pandas.DataFrame): The corpus to plot
        docs (int, list or slice): the documents to include in the plot, default: all documents
        features (int, list, or slice): the features to plot, default: top 50 features
        figsize (2-element list): size of the plot
    Returns:
        the plot
    """
    selection = corpus.ix.__getitem__((_prep_slice(docs), _prep_slice(features)))
    if figsize is None:
        w, h = plt.rcParams.get('figure.figsize')
        figsize = [1.5*w, 0.5*h]
    axes = selection.T.plot(kind='bar', figsize=figsize, **kwargs)
    return axes
