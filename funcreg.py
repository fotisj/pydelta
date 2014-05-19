"""
Simple function registry for pydelta.
"""

import itertools
import functools
import pandas as pd
import inspect
from collections import OrderedDict

class FunctionRegistry(OrderedDict):
    """
    A very simple function registry for delta functions
    """

    def __init__(self, name=None, comment="", config=None):
        super(FunctionRegistry, self).__init__()
        self.name = name
        self.comment = comment
        self.config = config

    def register(self, func):
        """
        Registers the given function with this registry.
        """
        name = func.__name__
        self[name] = func
        return func

    def get_title(self, func):
        """
        Returns the function's title, if available.

        `func` may be either a function or the name of a registered function.
        """
        if type(func) == 'string':
            func = self[func]

        return next(line for line in func.__doc__.splitlines() if line).strip()

    def needs_refcorpus(self, func):
        """
        Returns True if the function `func` needs a refcorpus argument.
        """
        if type(func) == 'string':
            func = self[func]
        sig = inspect.signature(func)
        return "refcorpus" in sig.parameters

    def _get_args(self, func):
        param = inspect.signature(func).parameters
        args = OrderedDict()
        for arg in param:
            if arg != "corpus" and arg != "refcorpus":
                args[arg] = param[arg]
        return args


def apply_distance_function(function, corpus, *args, **kwargs):
    """
    Applies the given distance function to the corpus.

    :param function: a distance function that takes two 1-dimensional vectors
    u, v and returns a number. 
    :param corpus: a dataframe mapping documents (columns) and words (index) to
    their respective frequency.
    
    Additional args and kwargs are passed on to the function. See also
    `delta_function` for an automatic factory that creates delta functions from
    distance functions.
    """
    deltas = pd.DataFrame(index=corpus.columns, columns=corpus.columns)
    for doc1, doc2 in itertools.combinations(corpus.columns, 2):
        delta = function(corpus[doc1], corpus[doc2], *args, **kwargs)
        deltas.at[doc1, doc2] = delta
        deltas.at[doc2, doc1] = delta
    return deltas.fillna(0)


def delta_function(function, *args, **kwargs):
    """Creates a delta function."""

    delta = functools.partial(apply_distance_function, function, 
            *args, **kwargs)
    functools.update_wrapper(delta, function)
    delta.distfunc = function
    return delta


