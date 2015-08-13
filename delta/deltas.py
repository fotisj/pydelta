# -*- encoding: utf-8 -*-
"""
This module contains the actual delta measures.

Normalizations
==============

A *normalization* is a function that works on a :class:`Corpus` and returns a
somewhat normalized version of that corpus. Each normalization has the
following additional attributes:

* name – an identifier for the normalization, usually the function name
* title – an optional, human-readable name for the normalization

Each normalization leaves its name in the 'normalizations' field of the corpus'
:class:`Metadata`.

All available normalizations need to be registered to the normalization
registry.


Delta Functions
===============

A *delta function* takes a :class:`Corpus` and creates a :class:`Distances`
table from that. Each delta function has the following properties:

* descriptor – a systematic descriptor of the distance function. For simple
    delta functions (see below), this is simply the name. For composite distance
    functions, this starts with the name of a simple delta function and is followed
    by a list of normalizations (in order) that are applied to the corpus before
    applying the distance function.
* name – a unique name for the distance function
* title – an optional, human-readable name for the distance function.


Simple Delta Functions
----------------------

Simple delta functions are functions that

"""

import logging
logger = logging.getLogger(__name__)

import pandas as pd
import scipy.spatial.distance as ssd
from scipy import linalg
from itertools import combinations
from functools import update_wrapper
from .util import Metadata
from .corpus import Corpus

sep = '-'   # separates parts of a descriptor

class _FunctionRegistry:
    """
    The registry of normalizations and delta functions.

    Usually, functions register themselves when they are created using one of the
    base classes or decorators (see below), they can be accessed using the registry's
    methods :meth:`normalization` and :meth:`delta`, or using subscription or
    attribute access.
    """

    def __init__(self):
        self.normalizations = {}    # name -> normalization function
        self.deltas = {}            # descriptor -> normalization function
        self.aliases = {}           # name -> normalization function

    @staticmethod
    def get_name(f):
        try:
            return f.name
        except:
            return f.__name__

    def add_normalization(self, f):
        """
        Registers the normalization _f_.

        Args:
            f (Normalization): The normalization to register.
        """
        name = self.get_name(f)
        if name in self.normalizations:
            logger.warning("Registering %s as %s, replacing existing function with this name", f, name)
        self.normalizations[name] = f

    def add_delta(self, f):
        """
        Registers the given Delta function.

        Args:
            f (DeltaFunction): The delta function to register.
        """
        self.deltas[f.descriptor] = f
        if f.name != f.descriptor:
            self.aliases[f.name] = f


    def normalization(self, name):
        """
        Returns the normalization identified by the name, or raises an :class:`IndexError` if
        it has not been registered.

        Args:
            name (str): The name of the normalization to retrieve.
        Returns:
            Normalization
        """
        return self.normalizations[name]

    def delta(self, descriptor, register=False):
        """
        Returns the delta function identified by the given descriptor or alias.
        If you pass in a composite descriptor and the delta function has _not_
        been registered yet, this tries to create a
        :class:`CompositeDeltaFunction` from the descriptor.

        Args:
            descriptor (str): Descriptor for the delta function to retrieve or create.
            register (bool): When creating a composite delta function,register
                it for future access.
        Returns:
            DeltaFunction: The requested delta function.
        """
        if descriptor in self.deltas:
            return self.deltas[descriptor]
        elif descriptor in self.aliases:
            return self.aliases[descriptor]
        elif sep in descriptor:
            return CompositeDeltaFunction(descriptor, register=register)
        else:
            raise IndexError("Delta function '%s' has not been found, "
                    "and it is not a composite descriptor, either." % descriptor)

    def __getitem__(self, index):
        try:
            return self.normalization(index)
        except KeyError:
            return self.delta(index)

    def __getattr__(self, index):
        try:
            return self[index]
        except IndexError as error:
            raise AttributeError(error)

    def __dir__(self):
        attributes = list(super().__dir__())
        attributes.extend(self.normalizations.keys())
        attributes.extend((name for name in self.deltas.keys() if name.isidentifier()))
        attributes.extend(self.aliases.keys())
        return attributes

    def __str__(self):
        return \
        """
        {} Delta Functions:
        ------------------
        {}

        {} Normalizations:
        -----------------
        {}
        """.format(len(self.deltas),
                '\n'.join(str(d) for d in self.deltas.values()),
                len(self.normalizations),
                '\n'.join(str(n) for n in self.normalizations.values()))


registry = _FunctionRegistry()

class Normalization:
    """
    Wrapper for normalizations.
    """

    def __init__(self, f, name=None, title=None, register=True):
        self.normalize = f
        if name is None:
            name = f.__name__
        if title is None:
            title = name
        self.name = name
        self.title = title
        update_wrapper(self, f)

        if register:
            registry.add_normalization(self)

    def __call__(self, corpus, *args, **kwargs):
        return Corpus(self.normalize(corpus, *args, **kwargs),
                document_describer=corpus.document_describer,
                metadata=corpus.metadata, normalization=(self.name,))

    def __str__(self):
        result = self.name
        if self.title != self.name:
            result += ' ('+self.title+')'
        # add docstring?
        return result

def normalization(*args, **kwargs):
    """
    Decorator that creates a :class:`Normalization` from a function or
    (callable) object. Can be used without or with keyword arguments:

        name (str): Name (identifier) for the normalization. By default, the function's name is used.
        title (str): Human-readable title for the normalization.
    """
    name = kwargs['name']   if 'name' in kwargs  else None
    title = kwargs['title'] if 'title' in kwargs else None

    def createNormalization(f):
        return Normalization(f, name=name, title=title)
        # FIXME functools.wrap?
    if args and callable(args[0]):
        return createNormalization(args[0])
    else:
        return createNormalization

class DeltaFunction:
    """
    Abstract base class of a delta function.

    To define a delta function, you have various options:

    1. subclass DeltaFunction and override its :meth:`__call__` method with something that directly handles a :class:`Corpus`.
    2. subclass DeltaFunction and override its :meth:`distance` method with a distance function
    3. instantiate DeltaFunction and pass it a distance function, or use the :func:`delta` decorator
    4. use one of the subclasses
    """

    def __init__(self, f=None, descriptor=None, name=None, title=None, register=True):
        """
        Creates a custom delta function.

        Args:
            f (function): a distance function that calculates the difference
                between two feature vectors and returns a float. If passed,
                this will be used for the implementation.
            name (str): The name/id of this function. Can be inferred from
                `f` or `descriptor`.
            descriptor (str): The descriptor to identify this function.
            title (str): A human-readable title for this function.
            register (bool): If true (default), register this delta function
                with the function registry on instantiation.
        """
        if f is not None:
            if name is None:
                name = f.__name__
            self.distance = f
            update_wrapper(self, f)

        if name is None:
            if descriptor is None:
                name = type(self).__name__
            else:
                name = descriptor

        if descriptor is None:
            descriptor = name

        if title is None:
            title = name

        self.name = name
        self.descriptor = descriptor
        self.title = title
        logger.debug("Created a %s with name=%s, descriptor=%s, title=%s",
                type(self), name, descriptor, title)
        if register:
            self.register()

    def __str__(self):
        result = self.name
        if self.title != self.name:
            result += ' ('+self.title+')'
        if self.descriptor != self.name:
            result += ' = ' + self.descriptor
        return result

    @staticmethod
    def distance(u, v, *args, **kwargs):
        """
        Calculate a distance between two feature vectors.

        This is an abstract method, you must either inherit from DeltaFunction
        and override distance or assign a function in order to use this.

        Args:
            u, v (pandas.Series): The documents to compare.
            *args, **kwargs: Passed through from the caller
        Returns:
            float: Distance between the documents.
        Raises:
            NotImplementedError if no implementation is provided.
        """
        raise NotImplementedError("You need to either override DeltaFunction"
                                  "and override distance or assign a function"
                                  "to distance")

    def register(self):
        """Registers this delta function with the global function registry."""
        registry.add_delta(self)

    def iterate_distance(self, corpus, *args, **kwargs):
        """
        Calculates the distance matrix for the given corpus.

        The default implementation will iterate over all pairwise combinations
        of the documents in the given corpus and call :meth:`distance` on each
        pair, passing on the additional arguments.

        Clients may want to use :meth:`__call__` instead, i.e. they want to call
        this object as a function.

        Args:
            corpus (Corpus): feature matrix for which to calculate the distance
            *args, **kwargs: further arguments for the matrix
        Returns:
            pandas.DataFrame: square dataframe containing pairwise distances.
                The default implementation will return a matrix that has zeros
                on the diagonal and the lower triangle a mirror of the upper
                triangle.
        """
        df = pd.DataFrame(index=corpus.index, columns=corpus.index)
        for a, b in combinations(df.index, 2):
            delta = self.distance(corpus[a], corpus[b], *args, **kwargs)
            df[a, b] = delta
            df[b, a] = delta
        return df.fillna(0)

    def create_result(self, df, corpus):
        """
        Wraps a square dataframe to a DistanceMatrix, adding appropriate
        metadata from corpus and this delta function.

        Args:
            df (pandas.DataFrame): Distance matrix like created by :meth:`iterate_distance`
            corpus (Corpus): source feature matrix
        Returns:
            DistanceMatrix: df as values, appropriate metadata
        """
        return DistanceMatrix(df, corpus.metadata, corpus=corpus,
                              delta=self.name,
                              delta_descriptor=self.descriptor)

    def __call__(self, corpus):
        """
        Calculates the distance matrix.

        Args:
            corpus (Corpus): The feature matrix that is the basis for the distance
        Returns:
            DistanceMatrix: Pairwise distances between the documents
        """
        return self.create_result(self.iterate_distance(corpus), corpus)

class CompositeDeltaFunction(DeltaFunction):
    """
    A composite delta function consists of a *basis* (which is another delta
    function) and a list of *normalizations*. It first transforms the corpus
    via all the given normalizations in order, and then runs the basis on the
    result.
    """

    def __init__(self, descriptor, name=None, title=None, register=True):
        """
        Creates a new composite delta function.

        Args:
            descriptor (str): Formally defines this delta function. First the
                name of an existing, registered distance function, then, separated
                by ``-``, the names of normalizations to run, in order.
            name (str): Name by which this delta function is registered, in
                addition to the descriptor
            title (str): human-readable title
            register (bool): If true (the default), register this delta
                function on creation
        """
        items = descriptor.split(sep)
        self.basis = registry.deltas[items[0]]
        del items[0]
        self.normalizations = [registry.normalizations[n] for n in items]
        super().__init__(self, descriptor, name, title, register)

    def __call__(self, corpus):
        for normalization in self.normalizations:
            corpus = normalization(corpus)
        return self.create_result(self.basis(corpus), corpus)


class PDistDeltaFunction(DeltaFunction):
    """
    Wraps one of the metrics implemented by :func:`ssd.pdist` as a delta function.
    """
    def __init__(self, metric, name=None, title=None, register=True, **kwargs):
        """
        Args:
            metric (str):  The metric that should be called via ssd.pdist
            name (str):    Name / Descriptor for the delta function, if None, metric is used
            title (str):   Human-Readable Title
            register (bool): If false, don't register this with the registry
            **kwargs:      passed on to :func:`ssd.pdist`
        """
        self.metric = metric
        self.kwargs = kwargs
        if name is None:
            name = metric
        if title is None:
            title = name.title() + " Distance"

        super().__init__(descriptor=name, name=name, title=title, register=register)

    def __call__(self, corpus):
        return self.create_result(pd.DataFrame(index=corpus.index, columns=corpus.index,
                data=ssd.squareform(ssd.pdist(corpus, self.metric, self.kwargs))), corpus)


class DistanceMatrix(pd.DataFrame):
    """
    A distance matrix is the result of applying a :class:`DeltaFunction` to a
    :class:`Corpus`.
    """

    def __init__(self, df, metadata, corpus=None, document_describer=None, **kwargs):
        super().__init__(df)
        self.metadata = Metadata(metadata, **kwargs)
        if document_describer is not None:
            self.document_describer = document_describer
        elif corpus is not None:
            self.document_describer = corpus.document_describer
        else:
            self.document_describer = None


    @classmethod
    def from_csv(cls, filename):
        """
        Loads a distance matrix from a cross-table style csv file.
        """
        df = pd.DataFrame.from_csv(filename)
        md = Metadata.load(filename)
        return cls(df, md)

    def save(self, filename):
        self.to_csv(filename)
        self.metadata.save(filename)


################# Now a bunch of normalizations:

@normalization(title="Z-Score")
def z_score(corpus):
    """Normalizes the corpus to the z-scores."""
    return (corpus - corpus.mean()) / corpus.std()

@normalization
def eder_std(corpus):
    """
    Returns a copy of this corpus that is normalized using Eder's normalization.
    This multiplies each entry with :math:`\frac{n-n_i+1}{n}`
    """
    n = corpus.columns.size
    ed = pd.Series(range(n, 0, -1), index=corpus.columns) / n
    return corpus.apply(lambda f: f*ed, axis=1)

@normalization
def binarize(corpus):
    """
    Returns a copy of this corpus in which the word frequencies are
    normalized to be either 0 (word is not present in the document) or 1.
    """
    df = corpus.copy()
    df[df > 0] = 1
    metadata = corpus.metadata
    del metadata["frequencies"]
    metadata["binarized"] = True
    return Corpus(corpus=df, metadata=metadata)

@normalization
def length_normalized(corpus):
    """
    Returns a copy of this corpus in which the frequency vectors
    have been length-normalized.
    """
    return corpus / corpus.apply(linalg.norm)

@normalization
def diversity_scaled(corpus):
    """
    Returns a copy of this corpus which has been scaled by the diversity argument from the Laplace distribution.
    """
    def diversity(values):
        return (values - values.median()).abs().sum() / values.size
    return corpus / corpus.apply(diversity)

@normalization
def sqrt(corpus):
    return corpus.sqrt()


################ Here come the deltas

PDistDeltaFunction("cityblock", "manhattan", title="Manhattan Distance")
PDistDeltaFunction("euclidean")
PDistDeltaFunction("cosine")
PDistDeltaFunction("canberra")
PDistDeltaFunction("braycurtis", title="Bray-Curtis Distance")
PDistDeltaFunction("correlation")
PDistDeltaFunction("chebyshev")

CompositeDeltaFunction("manhattan-z_score", "burrows", "Burrows' Delta")
CompositeDeltaFunction("manhattan-diversity_scaled", "linear", "Linear Delta")
CompositeDeltaFunction("euclidean-z_score", "quadratic", "Quadratic Delta")
CompositeDeltaFunction("manhattan-z_score-eder_std", "eder", "Eder's Delta")
CompositeDeltaFunction("manhattan-sqrt", "eder_simple", "Eder's Simple")
CompositeDeltaFunction("cosine-z_score", "cosine_delta", "Cosine Delta")

# TODO hoover # rotated # pielström

