# -*- coding: utf-8 -*-

"""
pydelta library
---------------

Stylometrics in Python
"""

__title__ = 'delta'
__version__ = '2.0.0'
__author__ = 'Fotis Jannidis, Thorsten Vitt'

from .corpus import Corpus, FeatureGenerator
from .deltas import registry, Normalization, DeltaFunction, \
        PDistDeltaFunction, CompositeDeltaFunction
from .cluster import Clustering

__all__ = [ Corpus, FeatureGenerator,
           registry, Normalization,
           DeltaFunction, PDistDeltaFunction, CompositeDeltaFunction, Clustering ]
