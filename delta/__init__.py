# -*- coding: utf-8 -*-

"""
pydelta library
---------------

Stylometrics in Python
"""

__title__ = 'delta'
__version__ = '2.0.0'
__author__ = 'Fotis Jannidis, Thorsten Vitt'

from delta.corpus import Corpus, FeatureGenerator, LETTERS_PATTERN, WORD_PATTERN
from delta.deltas import registry, Normalization, DeltaFunction, \
        PDistDeltaFunction, CompositeDeltaFunction
from delta.cluster import Clustering, KMedoidsClustering, FlatClustering
from delta.features import get_rfe_features
from delta.graphics import Dendrogram

__all__ = [ Corpus, FeatureGenerator, LETTERS_PATTERN, WORD_PATTERN,
           registry, Normalization,
           DeltaFunction, PDistDeltaFunction, CompositeDeltaFunction,
           Clustering, FlatClustering,  KMedoidsClustering,
           get_rfe_features, Dendrogram ]
