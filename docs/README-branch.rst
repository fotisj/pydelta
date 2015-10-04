--------
OBSOLETE
--------


Method Evaluation Branch
========================

This branch contains a bunch of scripts that we use to evaluate the distance
measures against various combinations of word count, corpus and case. It will
eventually be refactored to be less of a mess and merged back into the master
branch, meanwhile, here's an overview on what is what and how to use it.

All scripts can be called with a ``--help`` parameter and they will display an
up-to-date usage page with a short description of the available options.


delta.py
--------

The core of the delta implementation. It can both be used as a module or as
a command line script that is configured using a configuration file,
``pydelta.ini``. When used as a script, it performs one attribution experiment
that is configured using the configuration file, it can produce, show and save
a dendrogram and all kinds of intermediate results.

evaluate.py
-----------

This script will read a corpus and optionally a reference corpus. It will then
run a three-fold nested loop over distance measure, word count, and case
sensitivity, and calculate a distance matrix for each combination. Each
distance matrix is saved to a csv file named
``Distance_Measure.mfws.case_sensitivity.csv`` in the target directory.

The script needs to be passed a corpus directory that contains utf-8 encoded text files whose filenames must follow the pattern ``Author_Title.txt``. 

The generated csv files will be written as a cross table, i.e. first row and
first column are filenames from the corpus and the other cells contain the difference between the respective files.


concatenate-deltas.py
---------------------

This script can read directories full of tables written by ``evaluate.py`` and
perform three tasks:

- it can transform the “cross tables” a long format and concatenate them to a long form with the following columns:

    - File1
    - File2
    - Difference
    - Algorithm
    - Words
    - Case_Sensitive
    - Corpus    (from the directory name)
    - Author1
    - Title1
    - Author2
    - Title2

These long-format tables can get quite large. They will be saved as CSV, but can optionally be pickled instead.

- it optionally (``-e``) calculates three scores for each distance matrix: the difference of the means of the s standardized distances for same and different authors in the corpus, and two variatons of a clustering error according to a hierarchical clustering of the results

- it can optionally paint an (ugly) dendrogram (``-d``) for each distance matrix.

plot-scores.py
--------------

A very simple script that takes the score csvs written by concatenate-deltas.py (``-e`` option) and creates a bunch of plots from them.
