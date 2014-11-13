delta core
==========

Corpus
------

- Corpus inherits from DataFrame, which overrides the attribute subscription, which makes it hard to introduce attributes, which makes passing in options really really clumsy
  - from the discussions on gh etc., there are two basic problems:
    - nearly all operations on DFs return a new DF and these do not copy additional attributes or metadata,
    - using the df constructor is hardcoded in many places, so operations on our class will
      return just a plain DF and not an instance of our class.
  - there are some attempts to resolve that problem, e.g., by composition
- Tokenizing and calculating the frequency table is not really separated, making it hard to both use the tokenizer separately and replacing the tokenizer
- Old implementation lost the absolute word count. The new implementation however may contain either word counts or word frequencies, and it uses an heuristic to find out what it does. Additionally, while it is sorted after _tokenizing_ it is now not clear whether it is sorted after loading a corpus from a file.
- For some deltas it would be useful to access a z-scored version of the frequencies

Improvements:

- Sort out the local attribute stuff. Either find a way to access attributes, or use aggregation instead of inheritance for the dataframe.
- Store metadata, e.g., language, but also whether we store word counts or frequencies, whether we are sorted or culled
- Extract the tokenizer
- implement zscore(), which returns a zscored version of the corpus


Delta / Distance Methods
------------------------

- we need a way to represent a distance matrix with the metadata that was used to generate it. Probably best to implement this in Delta and save the experiment metadata there. We would then also need a way (factory/constructor) to load a distance matrix (w/metadata) from a file instead of calculating it.
- the ``const`` stuff is quite complicated to use. Iterating over algorithms requires access to ``const.__dict__``, adding an algorithm requires work at at least three places (add implementation, add ``elif`` clause, add constant value and increment constant counter), ``const`` is an implicit name
- we cannot clearly specify whether an algorithm requires a reference corpus

Evaluate
--------

- re-implement the cluster errors with the helper functions from scipy.cluster.hierarchy instead of the dendrogram


Figure
------

- Figure code is really really hard to use from outside. It has way to many constructor arguments, it requires a certain order of calls, and OTOH too much is hardcoded, e.g., filetype and filenames and text in image.

Improvements:

- Check for each filename transformation method whether it must be a method
- extract clustering from that class. Maybe lazily call it if neccessary.
- experiment metadata (mfw etc.) should be stored with the experiment and extracted from there
- titles should be configurable
- can we just store the Figure from ``plt.gcf()`` and work on that? Extract code to work out the figure from ``show()`` so we can paint and then save the dendrogram and leave show() for the interactive use. 
- make save() configurable, or let the user just access the thing.
