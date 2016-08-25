# -*- encoding: utf-8 -*-
"""
Various visualization tools.
"""

import logging
logger = logging.getLogger(__name__)

import scipy.cluster.hierarchy as sch
import matplotlib as mpl
import matplotlib.pyplot as plt
# from scipy import linalg
# from scipy.misc import comb
# from itertools import combinations
# from functools import update_wrapper


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
