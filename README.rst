-------
PyDelta
-------


pydelta is a library and tool to perform stylometric analyses like authorship attribution, and to evaluate methods for that. It originated as a commandline tool which implements three algorithms in the form described by Argamon in a paper on John Burrows Delta.
(S. Argamon, "Interpreting Burrows’s Delta: Geometric and Probabilistic 
Foundations," Literary and linguistic computing, vol. 23, iss. 2, pp. 131-147, 2008.)

Delta is a measure to describe the stylistic difference between texts. It is used
in computational stylistics, especially in author attribution. 
This implementation is for research purposes only, If you want to use
a reliable implementation with a nice Gui and much more features you should 
have a closer look at the great R tool 'stylo': 
https://sites.google.com/site/computationalstylistics/


Installation and Requirements
=============================

PyDelta requires **Python 3.3 or newer**. It has quite a set of dependencies (NumPy, Pandas, SciPy, SciKit-Learn, …), but it comes with a setup script that installs it with its dependencies.

Developers can clone the repo and run pip to install::

    git clone -b next https://github.com/fotis007/pydelta
    cd pydelta
    pip install -r requirements.txt

This will also install a version of scikit-learn that features
k-Medoids-Clustering which is not in the official version yet.

If you only wish to use it, not to hack on it, pip can clone and install it for you::

    pip install git+https://github.com/fotis007/pydelta@next

(Note that the Python 3 version of pip is sometimes called ``pip3``).


Usage
=====

There is no command line script yet. Read on in the Concepts guide to get started.

----

Thanks go to

- Thorsten Vitt for his help with profiling some critical parts and general improvements
- Allan Riddell for advice on matplotlib
