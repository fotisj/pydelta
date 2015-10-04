--------
Concepts
--------


pydelta is a library and tool to perform stylometric analyses like authorship attribution, and to evaluate methods for that. This document gives an overview of the concepts and data structures of this tool, read the API documentation for details.

Steps of an Experiment
======================

To perform a single experiment, we first extract features from a corpus to extract a raw **feature matrix**. This feature matrix is then postprocessed, e.g., by reducing or normalizing the features. On the final postprocessed feature matrix we run a **delta function** which will produce a **distance matrix**, with distances between the documents in the feature matrix. The distance matrix is then the basis for a **clustering**. The currently implemented clustering mechanism produces a hierarchical clustering, which can then be postprocessed to produce a flat clustering.

1. Feature Matrix
-----------------

A *feature matrix* describes the original corpus in tabular form: Each row represents a document, each column a feature, and each table cell contains a numeric value indicating the “signal strength” of the respective feature in the respective document. A *raw* feature matrix contains absolute frequencies in the cells, i.e. if our features are word frequencies, there might be a ``4196`` in the *und* column of the *Agathon* row to indicate that the document *Agathon* contains 4196 occurences of the word “und”.

1.1 Feature Extraction
^^^^^^^^^^^^^^^^^^^^^^

To create a feature matrix from some raw data, we use a :class:`FeatureGenerator`. A feature generator reads a document and produces a feature vector (a :class:`pandas.Series` of features) for that document. The :class:`FeatureMatrix` initializer reads a directory of files, passes each file through a :class:`FeatureGenerator` and assembles the resulting series to a :class:`pandas.DataFrame`. There is a default implementation of :class:`FeatureGenerator` that can parse text files, tokenizes them according to a configurable regular expression, and then returns a vector of word counts.

1.2 Feature Matrix Manipulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are some methods that can be used to manipulate the feature matrix. Each of these methods returns a new FeatureMatrix that is based on the current one, manipulated according to the given parameters. 

Some of these methods only work on absolute feature matrixes. Some of them return a feature matrix that is no longer absolute.


* `sort` sorts the features by the sum of the values for each document
* `reparse_column` replaces or adds a single column by running the feature generator with a given set of arguments.
* `relativize` converts absolute to relative frequencies
* `top_n` returns the top n features only. This runs sort unless the table has been sorted before.
* `cull` removes features that are only represented in few documents
* (`normalize` – you can also run one of the normalizations)

2. Delta Function
-----------------

A *delta function* transforms a feature matrix into a distance matrix. It usually does so by first applying zero or more *normalizations* to the feature matrix, and then running a distance function on the pairwise combinations of documents.

Both normalizations and delta functions are implemented as *callable objects* (that behave like standard objects as well as functions at the same time). There is a function registry that lists all available normalizations and delta functions, and we provide some tools to ease writing these.

Normalizations
^^^^^^^^^^^^^^

A normalization transforms a feature matrix in some way. The result is the same kind of feature matrix as before the normalization, however, the normalization will add to the matrix' metadata that it had run.

To create a normalization, you would usually write a simple function that works with data frames and apply the `normalization` decorator to that::

    @normalization
    def z_score(features):
        """Normalizes the feature matrix to the z-scores."""
        return (features - features.mean()) / features.std()

The decorator will wrap this in a :class:`Normlization` class and take care of dealing with the feature matrix metadata.


Delta Functions
^^^^^^^^^^^^^^^

A delta function transforms a feature matrix of n documents with m features into a n×n distance matrix of pairwise distances. There is a base class :class:`DeltaFunction` for all delta functions, and there are tools to make defining a new delta function as easy as possible.

Each delta function has a *descriptor* that uniquely identifies it. There may also be alias names and titles. Most often you will have to do with two kinds of delta function:

* a :class:`PDistDeltaFunction` is based on a specific distance function that is available via :function:`scipy.spatial.distance.pdist`
* a :class:`CompositeDeltaFunction` is created by first applying one or more normalizations to the distance matrix and then running a different delta function, e.g. a pdist based one.

The latter is created from a *descriptor* which consists of the distance function, followed by the normalizations in order::
 
    PDistDeltaFunction("cityblock", "manhattan", title="Manhattan Distance")
    CompositeDeltaFunction("manhattan-z_score", "burrows", "Burrows' Delta")


Running Multiple Experiments
============================

TODO
