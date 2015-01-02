"""
The delta.corpus module contains code for building, loading, saving, and manipulating
the representation of a corpus. Its heart is the :class:`Corpus` class which
represents the feature matrix. Also contained are default implementations for
reading and tokenizing files and creating a feature vector out of that.
"""

import os
import glob
import regex as re
import pandas as pd
import collections
import csv
from math import ceil
from .util import Metadata

import logging


class FeatureGenerator(object):
    """
    A **feature generator** is responsible for converting a subdirectory of files into a feature matrix (that will then become a corpus). If you need to customize the feature extraction process, create a custom feature generator and pass it into your :class:`Corpus` constructor call along with its `subdir` argument.

    The default feature generator is able to process a directory of text files, tokenize each of the text files according to a regular expression, and count each token type for each file. To customize feature extraction, you have two options:

        1. for simple customizations, just create a new FeatureGenerator and set the constructor arguments accordingly. Look in the docstring for :meth:`__init__` for details.
        2. in more complex cases, create a subclass and override methods as you see fit.

    On a feature generator passed in to :class:`Corpus`, only two methods will be called:

        * :meth:`__call__`, i.e. the object as a callable, to actually generate the feature vector,
        * :attr:`metadata` to obtain metadata fields that will be included in the corresponding corpus. 
    So, if you wish to write a completely new feature generator, you can ignore the other methods. 
    """

    def __init__(self, lower_case=False, encoding="utf-8", glob='*.txt',
            token_pattern=re.compile(r'\b\p{L}+?\b', re.WORD)):
        """
        Creates a customized default feature generator.

        :param bool lower_case: If ``True``, normalize all tokens to lower case before counting them.
        :param str encoding: The encoding to use when reading files.
        :param str glob: The pattern inside the subdirectory to find files.
        :param re.pattern token_pattern: The regular expression used to identify tokens. The default will find the shortest sequence of letters between two word boundaries (according to the simple word-boundary algorithm from _Unicode regular expressions_)
        """
        self.lower_case = lower_case
        self.encoding = encoding
        self.glob = glob
        self.token_pattern = token_pattern
        self.logger = logging.getLogger(__name__)

    def __repr__(self):
        return type(self).__name__ + '(' + \
                ', '.join( key+'='+repr(value) for key, value in self.__dict__.items() if key != 'logger' ) + \
                ')'

    def tokenize(self, lines):
        """
        Tokenizes the given lines.

        This method is called by :meth:`count_tokens`. The default
        implementation will return an iterable of all tokens in the given
        :param:`lines` that matches the :attr:`token_pattern`. 

        :param lines: Iterable of strings in which to look for tokens.
        :returns: Iterable (default implementation generator) of tokens
        """
        for line in lines:
            yield from self.token_pattern.findall(line)

    def count_tokens(self, lines):
        """
        This calls :meth:`tokenize` to split the iterable `lines` into tokens. If the
        :attr:`lower_case` attribute is given, the tokens are then converted to
        lower_case. The tokens are counted, the method returns a
        :class:`pd.Series` mapping each token to its number of occurrences. 

        This is called by :meth:`process_file`.

        :param lines: Iterable of strings in which to look for tokens.
        :returns: a :class:`pd.Series` mapping tokens to the number of occurrences.
        """
        # FIXME method name?
        if self.lower_case:
            tokens = (token.lower() for token in self.tokenize(lines))
        else:
            tokens = self.tokenize(lines)

        count = collections.defaultdict(int)
        for token in tokens:
            count[token] += 1
        return pd.Series(count)

    def get_name(self, filename):
        """
        Converts a single file name to a label for the corresponding feature vector.

        :rtype: str
        """
        return os.path.basename(filename).rsplit('.', 1)[0]

    def process_file(self, filename):
        """
        Processes a single file to a feature vector. 

        The default implementation reads the file pointed to by `filename` as a
        text file, calls :meth:`count_tokens` to create token counts and
        :meth:`get_name` to calculate the label for the feature vector.

        :param str filename: The path to the file to process
        :returns: A :class:`pd.Series` with feature counts, its name set according to :meth:`get_name`
        :rtype: pd.Series
        """
        self.logger.info("Reading %s ...", filename)
        with open(filename, "rt", encoding=self.encoding) as file:
            series = self.count_tokens(file)
            if series.name is None:
                series.name = self.get_name(filename)
            return series

    def process_directory(self, directory):
        """
        Iterates through the given directory and runs :meth:`process_file` for
        each file matching :attr:`glob` in there.

        :param str directory: Path to the directory to process
        :returns: a :class:`dict` mapping name to :class:`pd:Series`
        """
        filenames = glob.glob(os.path.join(directory, self.glob))
        if len(filenames) == 0:
            self.logger.error("No files matching %s in %s. Feature matrix will be empty.", self.glob, directory)
        else:
            self.logger.info("Reading %d files matching %s from %s", len(filenames), self.glob, directory)
        data = (self.process_file(filename) for filename in filenames)
        return { series.name : series for series in data }

    def __call__(self, directory):
        """
        Runs the feature extraction using :meth:`process_directory` for the
        given directory and returns a simple, unsorted pd.DataFrame for that.
        """
        df = pd.DataFrame(self.process_directory(directory))
        return df.T
    
    @property
    def metadata(self):
        """
        Returns metadata record that describes the parameters of the 
        features used for corpora created using this feature generator.

        :rtype: Metadata
        """
        return Metadata(features='words', lower_case=self.lower_case)


class Corpus(pd.DataFrame):

    def __init__(self, subdir=None, file=None, corpus=None, feature_generator=FeatureGenerator(),
            metadata=None, **kwargs):
        """
        Creates a new Corpus. 

        :param str subdir: Path to a subdirectory containing the (unprocessed) corpus data.
        :param str file: Path to a CSV file containing the feature vectors.
        :param pd.DataFrame corpus: A dataframe or :class:`Corpus` from which to create a new corpus, as a copy.
        :param FeatureGenerator feature_generator: A customizeable helper class that will process a `subdir` to a feature matrix, if the `subdir` argument is also given.
        :param dict metadata: A dictionary with metadata to copy into the new corpus.
        :param **kwargs: Additional keyword arguments will be set in the metadata record of the new corpus.
        """
        logger = logging.getLogger(__name__)

        # normalize the source stuff
        if subdir is not None:
            if isinstance(subdir, pd.DataFrame):
                subdir, corpus = None, subdir
            elif os.path.isfile(subdir):
                subdir, file = None, subdir
        
        # initialize or update metadata
        if metadata is None:
            metadata = Metadata(
                ordered=False,
                words=None,
                corpus=subdir if subdir else file,
                frequencies=False)
        else:
            metadata = Metadata(metadata) # copy it, just in case

        # initialize data
        if subdir is not None:
            logger.info("Creating corpus by reading %s using %s", subdir, feature_generator)
            df = feature_generator(subdir)
            metadata.update(feature_generator)
        elif file is not None:
            logger.info("Loading corpus from CSV file %s ...", file)
            df = pd.read_csv(file, index_col=0).T
            try:
                metadata = Metadata.load(file)
            except OSError as e:
                self.logger.warning("Failed to load metadata for %s. Using defaults: %s", file, metadata, exc_info=True)
            # TODO can we probably use hdf5?
        elif corpus is not None:
            df = corpus
            if isinstance(corpus, Corpus):
                metadata.update(corpus.metadata)
        else:
            raise ValueError("Error. Only one of subdir and corpusfile can be not None")

        metadata.update(**kwargs)

        if not metadata.ordered:
            df = df.iloc[:,(-df.sum()).argsort()]
            metadata.ordered = True

        super().__init__(df)
        self.logger = logger
        self.metadata = metadata

    def save(self, filename="corpus_words.csv"):
        """
        saves corpus to file.
        """
        self.logger.info("Saving corpus to %s ...", filename)
        self.T.to_csv(filename, encoding="utf-8", na_rep=0, quoting=csv.QUOTE_NONNUMERIC)
        self.metadata.save(filename)
        # TODO different formats? compression?

    def get_mfw_table(self, mfwords):
        """
        Shortens the list to the given number of most frequent words and converts
        the word counts to frequencies

        This returns a new :class:`Corpus`, the data in this object is not modified.

        :param mfwords: number of most frequent words in the new corpus.
        :returns: a new sorted corpus shortened to `mfwords`
        """        
        new_corpus = self / self.sum() if not self.metadata.frequencies else self
        #slice only mfwords from total list
        if mfwords > 0:
            return Corpus(corpus=new_corpus.iloc[:,:mfwords], metadata=self.metadata, words=mfwords, frequencies=True)
        else:
            return Corpus(corpus=new_corpus, metadata=self.metadata, frequencies=True)


    def cull(self, ratio=None, threshold=None, keepna=False):
        """
        Performs culling, i.e. returns a new corpus with all words that do not
        appear in at least a given ratio or absolute number of documents
        removed.

        :param float ratio: Minimum ratio of documents a word must occur in to
            be retained. Note that we're always rounding towards the ceiling,
            i.e.  if the corpus contains 10 documents and ratio=1/3, a word
            must occur in at least *4* documents
        :param int threshold: Minimum number of documents a word must occur in
            to be retained
        :param bool keepna: If set to True, the missing words in the returned
            corpus will be retained as ``nan`` instead of ``0``.
        :rtype: :class:`Corpus`
        """
        if ratio is not None:
            if ratio > 1:
                threshold = ratio
            else:
                threshold = ceil(ratio * self.index.size)
        elif threshold is None:
            return self

        culled = self.replace(0, float('NaN')).dropna(thresh=threshold, axis=1)
        if not keepna:
            culled = culled.fillna(0)
        return Corpus(corpus=culled, metadata=self.metadata, culling=threshold)

    @staticmethod
    def diversity(values):
        """
        calculates the spread or diversity (wikipedia) of a laplace distribution of values
        see Argamon's Interpreting Burrow's Delta p. 137 and
        http://en.wikipedia.org/wiki/Laplace_distribution
        couldn't find a ready-made solution in the python libraries

        :param values: a pd.Series of values
        """
        return (values - values.median()).abs().sum() / values.size

    def diversities(self):
        """
        calculate the 'spread' of word distributions assuming they are laplace dist.

        :returns: a :class:`pandas.Series` with the diversity for each word in the corpus
        """
        return self.apply(self.diversity)
