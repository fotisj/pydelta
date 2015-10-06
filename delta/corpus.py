"""
The delta.corpus module contains code for building, loading, saving, and
manipulating the representation of a corpus. Its heart is the :class:`Corpus`
class which represents the feature matrix. Also contained are default
implementations for reading and tokenizing files and creating a feature vector
out of that.
"""

import os
import glob
from fnmatch import fnmatch
import regex as re
import pandas as pd
import collections
import csv
from math import ceil
from .util import Metadata, DocumentDescriber, DefaultDocumentDescriber

import logging


class FeatureGenerator(object):

    """
    A **feature generator** is responsible for converting a subdirectory of
    files into a feature matrix (that will then become a corpus). If you need
    to customize the feature extraction process, create a custom feature
    generator and pass it into your :class:`Corpus` constructor call along with
    its `subdir` argument.

    The default feature generator is able to process a directory of text files,
    tokenize each of the text files according to a regular expression, and
    count each token type for each file. To customize feature extraction, you
    have two options:

        1. for simple customizations, just create a new FeatureGenerator and
           set the constructor arguments accordingly. Look in the docstring for
           :meth:`__init__` for details.
        2. in more complex cases, create a subclass and override methods as you
           see fit.

    On a feature generator passed in to :class:`Corpus`, only two methods will
    be called:

        * :meth:`__call__`, i.e. the object as a callable, to actually generate
            the feature vector,
        * :attr:`metadata` to obtain metadata fields that will be included in
            the corresponding corpus.

    So, if you wish to write a completely new feature generator, you can ignore
    the other methods.
    """

    def __init__(self, lower_case=False, encoding="utf-8", glob='*.txt',
                 skip=None,
                 token_pattern=re.compile(r'\b\p{L}+?\b', re.WORD),
                 max_tokens=None):
        """
        Creates a customized default feature generator.

        Args:
            lower_case (bool): if ``True``, normalize all tokens to lower case
                before counting them
            encoding (str): the encoding to use when reading files
            glob (str): the pattern inside the subdirectory to find files.
            skip (str): don't handle files that match this pattern
            token_pattern (re.Regex): The regular expression used to identify
                tokens. The default will find the shortest sequence of letters
                between two word boundaries (according to the simple
                word-boundary algorithm from *Unicode regular expressions*)
            max_tokens (int): If set, stop reading each file after that many words.
        """
        self.lower_case = lower_case
        self.encoding = encoding
        self.glob = glob
        self.skip = skip
        self.token_pattern = token_pattern
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)

    def __repr__(self):
        return type(self).__name__ + '(' + \
            ', '.join(key+'='+repr(value)
                      for key, value in self.__dict__.items() if key != 'logger') + \
            ')'

    def tokenize(self, lines):
        """
        Tokenizes the given lines.

        This method is called by :meth:`count_tokens`. The default
        implementation will return an iterable of all tokens in the given
        :param:`lines` that matches the :attr:`token_pattern`.

        Args:
            lines: Iterable of strings in which to look for tokens.

        Returns:
            Iterable (default implementation generator) of tokens
        """
        count = 0
        for line in lines:
            for token in self.token_pattern.findall(line):
                count += 1
                yield token
                if self.max_tokens is not None and count >= self.max_tokens:
                    return

    def count_tokens(self, lines):
        """
        This calls :meth:`tokenize` to split the iterable `lines` into tokens.
        If the :attr:`lower_case` attribute is given, the tokens are then
        converted to lower_case. The tokens are counted, the method returns a
        :class:`pd.Series` mapping each token to its number of occurrences.

        This is called by :meth:`process_file`.

        Args:
            lines: Iterable of strings in which to look for tokens.
        Returns:
            pandas.Series: maps tokens to the number of occurrences.
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
        Converts a single file name to a label for the corresponding feature
        vector.

        Returns:
            str: Feature vector label (filename w/o extension by default)
        """
        return os.path.basename(filename).rsplit('.', 1)[0]

    def process_file(self, filename):
        """
        Processes a single file to a feature vector.

        The default implementation reads the file pointed to by `filename` as a
        text file, calls :meth:`count_tokens` to create token counts and
        :meth:`get_name` to calculate the label for the feature vector.

        Args:
            filename (str): The path to the file to process
        Returns:
            :class:`pd.Series`: Feature counts, its name set according to
                :meth:`get_name`
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

        Args:
            directory (str): Path to the directory to process

        Returns:
            dict: mapping name to :class:`pd:Series`
        """
        filenames = glob.glob(os.path.join(directory, self.glob))
        if len(filenames) == 0:
            self.logger.error(
                "No files matching %s in %s. Feature matrix will be empty.",
                self.glob,
                directory)
        else:
            self.logger.info(
                "Reading %d files matching %s from %s",
                len(filenames),
                self.glob,
                directory)
        data = (self.process_file(filename)
                for filename in filenames
                if self.skip is None or not(fnmatch(filename, self.skip)))
        return {series.name: series for series in data}

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
        Returns:
            Metadata: metadata record that describes the parameters of the
                features used for corpora created using this feature generator.
        """
        return Metadata(features='words', lower_case=self.lower_case)


class CorpusNotAbsolute(Exception):
    def __init__(self, operation):
        super().__init__("{} not possible: Absolute frequencies required.")


class Corpus(pd.DataFrame):

    def __init__(self, subdir=None, file=None, corpus=None,
                 feature_generator=FeatureGenerator(),
                 document_describer=DefaultDocumentDescriber(),
                 metadata=None, **kwargs):
        """
        Creates a new Corpus.

        Args:
            subdir (str): Path to a subdirectory containing the (unprocessed) corpus data.
            file (str): Path to a CSV file containing the feature vectors.
            corpus (pandas.DataFrame): A dataframe or :class:`Corpus` from which to create a new corpus, as a copy.
            feature_generator (FeatureGenerator): A customizeable helper class that will process a `subdir` to a feature matrix, if the `subdir` argument is also given.
            metadata (dict): A dictionary with metadata to copy into the new corpus.
            **kwargs: Additional keyword arguments will be set in the metadata record of the new corpus.
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
            metadata = Metadata(metadata)  # copy it, just in case

        # initialize data
        if subdir is not None:
            logger.info(
                "Creating corpus by reading %s using %s",
                subdir,
                feature_generator)
            df = feature_generator(subdir)
            metadata.update(feature_generator)
        elif file is not None:
            logger.info("Loading corpus from CSV file %s ...", file)
            df = pd.read_csv(file, index_col=0).T
            try:
                metadata = Metadata.load(file)
            except OSError:
                self.logger.warning(
                    "Failed to load metadata for %s. Using defaults: %s",
                    file,
                    metadata,
                    exc_info=True)
            # TODO can we probably use hdf5?
        elif corpus is not None:
            df = corpus
            if isinstance(corpus, Corpus):
                metadata.update(corpus.metadata)
        else:
            raise ValueError(
                "Error. Only one of subdir and corpusfile can be not None")

        metadata.update(**kwargs)

        if not metadata.ordered:
            df = df.iloc[:, (-df.sum()).argsort()]
            metadata.ordered = True

        super().__init__(df.fillna(0))
        self.logger = logger
        self.metadata = metadata
        self.document_describer = document_describer

    def save(self, filename="corpus_words.csv"):
        """
        Saves the corpus to a CSV file.

        The corpus will be saved to a CSV file containing documents in the
        columns and features in the rows, i.e. a transposed representation.
        Document and feature labels will be saved to the first row or column,
        respectively.

        A metadata file will be saved alongside the file.


        Args:
            filename (str): The target file.
        """
        self.logger.info("Saving corpus to %s ...", filename)
        self.T.to_csv(
            filename,
            encoding="utf-8",
            na_rep=0,
            quoting=csv.QUOTE_NONNUMERIC)
        self.metadata.save(filename)
        # TODO different formats? compression?

    def is_absolute(self) -> bool:
        """
        Returns:
            bool: ``True`` if this is a corpus using absolute frequencies
        """
        return not(self.metadata.frequencies)

    def get_mfw_table(self, mfwords):
        """
        Shortens the list to the given number of most frequent words and converts
        the word counts to relative frequencies

        This returns a new :class:`Corpus`, the data in this object is not modified.

        TODO separate methods?

        Args:
            mfwords (int): number of most frequent words in the new corpus. 0 means all words.

        Returns:
            Corpus: a new sorted corpus shortened to `mfwords`
        """
        new_corpus = self / \
            self.sum() if not self.metadata.frequencies else self
        # slice only mfwords from total list
        if mfwords > 0:
            return Corpus(
                corpus=new_corpus.iloc[
                    :,
                    :mfwords],
                document_describer=self.document_describer,
                metadata=self.metadata,
                words=mfwords,
                frequencies=True)
        else:
            return Corpus(corpus=new_corpus,
                          document_describer=self.document_describer,
                          metadata=self.metadata, frequencies=True)

    def cull(self, ratio=None, threshold=None, keepna=False):
        """
        Removes all features that do not appear in a minimum number of
        documents.

        Args:
            ratio (float): Minimum ratio of documents a word must occur in to
                be retained. Note that we're always rounding towards the
                ceiling, i.e.  if the corpus contains 10 documents and
                ratio=1/3, a word must occur in at least *4* documents (if this
                is >= 1, it is interpreted as threshold)
            threshold (int): Minimum number of documents a word must occur in
                to be retained
            keepna (bool): If set to True, the missing words in the returned
                corpus will be retained as ``nan`` instead of ``0``.

        Returns:
            Corpus: A new corpus witht the culled words removed. The original
                corpus is left unchanged.
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
        return Corpus(corpus=culled,
                      document_describer=self.document_describer,
                      metadata=self.metadata, culling=threshold)

    def reparse(self, feature_generator, subdir=None, **kwargs):
        """
        Parse or re-parse a set of documents with different settings.

        This runs the given feature generator on the given or configured
        subdirectory. The feature vectors returned by the feature generator
        will replace or augment the corpus.

        Args:
            feature_generator (FeatureGenerator): Will be used for extracting
                stuff.
            subdir (str): If given, will be passed to the feature generator for
                processing. Otherwise, we'll use the subdir configured with
                this corpus.
            **kwargs: Additional metadata for the returned corpus.
        Returns:
            Corpus: a new corpus with the respective columns replaced or added.
                The current object will be left unchanged.
        Raises:
            CorpusNotAbsolute: if called on a corpus with relative frequencies
        """
        if not(self.is_absolute()):
            raise CorpusNotAbsolute('Replacing or adding documents')
        if subdir is None:
            if self.metadata.corpus is not None \
                    and os.path.isdir(self.metadata.corpus):
                subdir = self.metadata.corpus
        reparsed = feature_generator(subdir)
        df = pd.DataFrame(self, copy=True)
        for new_doc in reparsed.index:
            df.loc[new_doc, :] = reparsed.loc[new_doc, :]
        return Corpus(corpus=df, metadata=self.metadata, **kwargs)

    def tokens(self) -> pd.Series:
        """Number tokens by text"""
        if self.is_absolute():
            return self.sum(axis=1)
        else:
            raise CorpusNotAbsolute('Calculation on absolute numbers')

    def types(self) -> pd.Series:
        """Number of different features by text"""
        if self.is_absolute():
            return self.replace(0, float('NaN')).count(axis=1)
        else:
            raise CorpusNotAbsolute('Calculation on absolute numbers')

    def ttr(self) -> float:
        """
        Type/token ratio for the whole corpus.

        See also:
            https://en.wikipedia.org/wiki/Lexical_density
        """
        return self.types().sum() / self.tokens().sum()

    def ttr_by_text(self) -> pd.Series:
        """
        Type/token ratio for each text.
        """
        return self.types() / self.tokens()
