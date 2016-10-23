Pydelta v 0.1
=============

pydelta is a commandline tool which implements 3 algorithms in the form 
described by Argamon in a paper on John Burrows Delta.
(S. Argamon, "Interpreting Burrows’s Delta: Geometric and Probabilistic 
Foundations," Literary and linguistic computing, vol. 23, iss. 2, pp. 131-147, 2008.)

Delta is a measure to describe the stylistic difference between texts. It is used
in computational stylistics, especially in author attribution. 
This implementation is for research purposes only, If you want to use
a reliable implementation with a nice Gui and much more features you should 
have a closer look at the great R tool 'stylo': 
https://sites.google.com/site/computationalstylistics/

Which branch should I use?
--------------------------

* The *develop* branch contains the most up-to-date version of the
  classic command line application together with a bunch of extra
  scripts that can be used to perform mass experiments on delta
  measures. The scripts allow to systematically variate parameters
  like algorithm, number of features, text length etc. and generate
  comparison charts. This branch also includes the largest number of
  delta measures, however the extra scripts are quite a mess.
* The *master* branch is basically a version of the develop branch
  before we introduced all those confusing scripts, with minor
  maintenance patches.
* The *next* branch is a major work-in-progress rewrite that is
  intended to be used as a python library. It makes it easier to
  program against parts of the delta stuff, but it does not yet
  contain the ``.ini`` file driven main program and the parameter
  sweeping scripts.

Installation
------------

1. Install python 3.3 or 3.4 (newer versions of 3.x haven't been tested, older versions don't work)
2. Install dependencies: pandas, scipy, matplotlib, profig (all on pypi).
3. Download or checkout pydelta.py

(you can run ``pip install -r requirements.txt`` to get all dependencies, or in
general: from python_dir/scripts ``pip install library_name`` where
library_name is pandas etc.)


Usage of pydelta:
-----------------

The first time you run the script, use::

    pydelta.py -O files.ini:True

Thus you create a configuration file (pydelta.ini) which allows you to set most
parameters of the script without changing the script.

Normal use:

Put the text files, you want to analyze, into a subdirectory called 'corpus'
under the directory this script is living in. The filenames for the corpus
should have the format authorname_title.txt. The script assumes that the name
has the format surname, firstname but should survive other delimiter.

Start the script with::

    ./delta.py

on windows (assuming python.exe is in your path)::

    python delta.py

You can always use commandline parameters to override default settings
/ settings in the pydelta.ini::

    ./delta.py -O figure.title:"German Novelists around 1800" -O stat.mfwords:2500

(setting a title and the amount of most frequent words used for delta). Use
``delta -h`` for more information or look into the ini-file to find
explanations of all parameters.

Thanks go to

- Thorsten Vitt for his help with profiling some critical parts and general improvements
- Allan Riddell for advice on matplotlib
