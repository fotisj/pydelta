API documentation
=================

The end-user facing stuff from the submodules is also exported by the package, so you can call, e.g., ``delta.Corpus``.

delta.corpus module
-------------------

Represent, build and manipulate feature matrixes.

.. automodule:: delta.corpus
    :members:
    :undoc-members:
    :show-inheritance:

delta.deltas module
-------------------

Normalizations for feature matrixes, distance functions, and distance matrices.

This module contains utilities to create both normalizations  (that normalize a feature matrix in some way; cf. :class:`Normalization`) and delta functions (that calculate a distance matrix from a feature matrix). 

All normalizations and delta functions created in an application are available from the function registry exposed as variable ``registry``, see :class:`FunctionRegistry` for details on the interface.

The module also provides a standard set of delta functions and normalizations ready to use.

.. automodule:: delta.deltas
    :members:
    :undoc-members:
    :show-inheritance:

delta.cluster module
--------------------

.. automodule:: delta.cluster
    :members:
    :undoc-members:
    :show-inheritance:

delta.util module
-----------------------

.. automodule:: delta.util
    :members:
    :undoc-members:
    :show-inheritance:

delta.graphics module
---------------------

.. automodule:: delta.graphics
    :members:
    :undoc-members:
    :show-inheritance:

Module contents
---------------

.. automodule:: delta
    :members:
    :undoc-members:
    :show-inheritance:
