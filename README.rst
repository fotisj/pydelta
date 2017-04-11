-------
PyDelta
-------


pydelta is a library to perform stylometric analyses like authorship attribution, and to evaluate methods for that. It originated as a commandline tool which implements three algorithms in the form described by Argamon in a paper on John Burrows Delta.
(S. Argamon, "Interpreting Burrows’s Delta: Geometric and Probabilistic 
Foundations," Literary and linguistic computing, vol. 23, iss. 2, pp. 131-147, 2008.)

Delta is a measure to describe the stylistic difference between texts. It is used
in computational stylistics, especially in author attribution. 
This implementation is for research purposes only, If you want to use
a reliable implementation with a nice Gui and much more features you should 
have a closer look at the great R tool Stylo_.

.. _Stylo: https://sites.google.com/site/computationalstylistics/

Documentation
=============

Documentation_ for the _next_ branch is updated on each commit.

.. _Documentation: http://dev.digital-humanities.de/ci/job/pydelta-next/Documentation/index.html



Different branches, different versions
======================================

There are currently three different long-running branches:

1. The **master** branch contains a rather old version of pydelta capable of running single experiments with a bunch of delta measures. Will be replaced sometime soon.
2. The **develop** branch is based on the script in the master version, but it contains more delta methods as well as some very unpolished command line scripts that we used, e.g., in order to produce the results presented at DH 2015 in Sidney.
3. The **next** branch is more or less a re-implementation that is not yet feature complete. It does not contain a command-line executable, and some of the methods that did not perform well have been left out.


Installation and Requirements
=============================

PyDelta requires **Python 3.3 or newer**. It has quite a set of dependencies (NumPy, Pandas, SciPy, SciKit-Learn, …), but it comes with a setup script that installs it with its dependencies.

If you only wish to use it, not to hack on it, pip can clone and install it for you::

    pip install git+https://github.com/cophi-wue/pydelta@next

(Note that the Python 3 version of pip is sometimes called ``pip3``).

Developers can clone the repo and run pip to install::

    git clone -b next https://github.com/cophi-wue/pydelta
    cd pydelta
    pip install -r requirements.txt

This will also install a version of scikit-learn that features
k-Medoids-Clustering which is not in the official version yet. You may need to
install numpy, scipy, and Cython separately first.


----

Thanks go to

- Thorsten Vitt for his help with profiling some critical parts and general improvements
- Allan Riddell for advice on matplotlib
