# -*- coding: utf-8 -*-

"""
pydelta library
---------------

Stylometrics in Python
"""

__title__ = 'delta'
__version__ = '2.0.0'
__author__ = 'Fotis Jannidis, Thorsten Vitt'

from warnings import warn
from delta.corpus import Corpus, FeatureGenerator, LETTERS_PATTERN, WORD_PATTERN
from delta.deltas import registry, normalization, Normalization, DeltaFunction, \
        PDistDeltaFunction, CompositeDeltaFunction
from delta.cluster import Clustering, FlatClustering

from delta.features import get_rfe_features
from delta.graphics import Dendrogram

__all__ = [ Corpus, FeatureGenerator, LETTERS_PATTERN, WORD_PATTERN,
           registry, Normalization, normalization,
           DeltaFunction, PDistDeltaFunction, CompositeDeltaFunction,
           Clustering, FlatClustering,
           get_rfe_features, Dendrogram ]

try:
        from delta.cluster import KMedoidsClustering
        __all__.append(KMedoidsClustering)
except (ImportError, NameError):
        warn("KMedoidsClustering not available")
