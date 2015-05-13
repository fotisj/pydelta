#!/usr/bin/env python3

"""
The experiments module can be used to perform a series of experiments in which
you vary some of the arguments. Here's the basic data model:

A _Facet_ is an aspect you wish to vary, e.g. the number of features. A facet
delivers a set of _expressions_. Each expression represents the actual values
of the facet, eg, "3000 most frequent words" might be an expression of the
facet 'number of features'.



There are some different kinds of facets:

A _corpus builder facet_ determines how the actual corpus is built. The corpus
builder facets are used to actually assemble a constructor call to the
:class:`delta.Corpus` class, i.e. for every combination of expressions we get a
new Corpus. Thus, variation here may be quite lengthy.

A _corpus manipulation facet_ takes an existing corpus and manipulates it, e.g., by
extracting the n most frequent words. This is much faster then building the
corpus anew each time, so if you can, implement a corpus manipulation facet instead
of a corpus builder one.

A _method facet_ delivers a delta function that should be manipulated.
"""
