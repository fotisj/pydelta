#!/usr/bin/env python3
"""
calculates Burrow's Delta and Argamon's proposed variations
tbd:
- add a more solid evaluation for the results.
- rotated_delta: convert the complex results, use abs()?
- load deltas from file (to check our results against stylo)
- replace print statements with logging mechanism?
- redo the saving of results in a more organized manner
"""
import sys
import glob
import re
import collections
import os
import csv
import itertools
from datetime import datetime
import argparse

import pandas as pd
import numpy as np
from scipy import linalg
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import matplotlib.pylab as plt
import profig


class Corpus(pd.DataFrame):
    def __init__(self, subdir=None, corpus=None, encoding="utf-8", limit=2000, set_limit=False, lower_case=False):
        if subdir is None and corpus is None:
            super().__init__(pd.read_csv("corpus.csv", index_col=0))
        elif subdir is not None and corpus is None:
            super().__init__(self.process_files(subdir, encoding, limit, set_limit, lower_case))
        elif corpus is not None:
            super().__init__(corpus)
        else:
            raise ValueError("Error. Only one of subdir and corpusfile can be not None")

    def process_files(self, subdir, encoding, limit, set_limit, lower_case):
        """
        preprocessing all files ending with *.txt in corpus subdir
        all files are tokenized
        a table of all word and their freq in all texts is created
        format
                filename1 filename2 filename3
        word      nr        nr
        word      nr        nr
        :param: config: access to configuration settings
        :param: filter: if defined, return only those words that are in the given list
        """
        if not os.path.exists(subdir):
            raise Exception("The directory " + subdir + " doesn't exist. \nPlease add a directory " +
                            "with text files.\n")
        filelist = glob.glob(subdir + os.sep + "*.txt")
        list_of_wordlists = []
        for file in filelist:
            list_of_wordlists.append(self.tokenize_file(file, encoding, limit, set_limit, lower_case))
        return pd.DataFrame(list_of_wordlists).fillna(0).T

    def tokenize_file(self, filename, encoding, limit, set_limit, lower_case):
        """
        tokenizes file and returns an unordered pandas.DataFrame
        containing the words and frequencies
        standard encoding = utf-8
        """
        all_words = collections.defaultdict(int)

        #read file, tokenize it, count words
        read_text_length = 0
        #reading the config information only once because of speed
        with open(filename, "r", encoding=encoding) as filein:
            print("processing " + filename)
            for line in filein:
                if set_limit:
                    if read_text_length > limit:
                        break
                    else:
                        read_text_length += len(line)
                words = re.findall("\w+", line)
                for w in words:
                    if lower_case:
                        w = w.lower()
                    all_words[w] += 1
        filename = os.path.basename(filename)
        wordlist = pd.Series(all_words, name=filename)
        return wordlist / wordlist.sum()

    def save(self):
        """
        saves corpus to file
        """
        print("Saving corpus to file corpus_words.csv")
        self.to_csv("corpus_words.csv", encoding="utf-8", na_rep=0, quoting=csv.QUOTE_NONNUMERIC)

    def get_mfw_table(self, mfwords):
        """
        sorts the table containing the frequency lists
        by the sum of all word freq
        returns the corpus list shortened to the most frequent words
        number defined by mfwords
        """
        #nifty trick to get it sorted according to sum
        #not from me :-)
        new_corpus = self.loc[(-self.sum(axis=1)).argsort()]
        #slice only mfwords from total list
        if mfwords > 0:
            return Corpus(corpus=new_corpus[:mfwords])
        else:
            return Corpus(corpus=new_corpus)

    def stds(self):
        """calculates std for all words of the corpus
           returns a pd.Series containing the means and the
           words as index"""
        return self.std(axis=1)

    def medians(self):
        """calculates medians for all words of the corpus
           returns a pd.Series containing the medians and the
           words as index"""
        return self.corpus.median(axis=1)

    def diversity(self, values):
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
        """
        return self.apply(self.diversity, axis=1)


class Delta():
    results = pd.DataFrame()

    def __init__(self, corpus, delta_choice, refcorpus=None):
        """
        chooses the algorithm for the calculation of delta
        """
        # XXX use functions directly & set a title attribute on them?
        if delta_choice == 0:
            self.results = self.classic_delta(corpus)
        elif delta_choice == 1:
            self.results = self.linear_delta(corpus)
        elif delta_choice == 2:
            self.results = self.quadratic_delta(corpus)
        elif delta_choice == 3:
            self.results = self.rotated_delta(corpus, refcorpus).abs()
        else:
            raise Exception("ERROR: You have to choose an algorithm for Delta.")

    def classic_delta(self, corpus):
        """
        calculates Delta in the simplified form proposed by Argamon
        """
        stds = corpus.stds()
        deltas = pd.DataFrame(index=corpus.columns, columns=corpus.columns)
        for i, j in itertools.combinations(corpus.columns, 2):
            delta = ((corpus[i] - corpus[j]).abs() / stds).sum() / len(corpus.index)
            deltas.at[i, j] = delta
            deltas.at[j, i] = delta
        return deltas.fillna(0)

    def quadratic_delta(self, corpus):
        """
        Argamon's quadratic Delta
        """
        vars_ = corpus.stds() ** 2
        deltas = pd.DataFrame(index=corpus.columns, columns=corpus.columns)
        for i, j in itertools.combinations(corpus.columns, 2):
            delta = ((corpus[i] - corpus[j]) ** 2 / vars_).sum()
            deltas.at[i, j] = delta
            deltas.at[j, i] = delta
        return deltas.fillna(0)

    def linear_delta(self, corpus):
        """
        Argamon's linear Delta
        """
        diversities = corpus.diversities()
        deltas = pd.DataFrame(index=corpus.columns, columns=corpus.columns)
        for i, j in itertools.combinations(corpus.columns, 2):
            delta = ((corpus[i] - corpus[j]).abs() / diversities).sum()
            deltas.at[i, j] = delta
            deltas.at[j, i] = delta
        return deltas.fillna(0)


    # rotated quadratic delta. This could be moved into the main delta function
    # when it is finished

    def _cov_matrix(self, corpus):
        # There's also pd.DataFrame.cov, which calculates the unbiased covariance
        # (normalized by n-1), but is much faster. XXX evaluate whether it would be
        # problematic to use that instead of our own _cov_matrix
        """
        Calculates the covariance matrix S consisting of the covariances $\sigma_{ij
        }$ for the words $w_i, w_j$ in the given comparison corpus.

        :param corpus: is a words x texts DataFrame representing the reference
        corpus.
        """
        means = corpus.mean(axis=1)
        documents = corpus.columns.size
        result = pd.DataFrame(index=corpus.index, columns=corpus.index, dtype=np.float)
        dev = corpus.sub(means, axis=0)
        for w in corpus.index:
            result.at[w, w] = (dev.loc[w] ** 2).sum() / documents

        # FIXME hier ist noch Optimierungspotential,
        # bei 2000 Wörtern läuft das ewig.
        # Ggf. corpus.loc[w]-means.at[w] cachen? ob's das bringt?
        for i, j in itertools.combinations(corpus.index, 2):
            cov = (dev.loc[i] * dev.loc[j]).sum() / documents
            result.at[i, j] = cov
            result.at[j, i] = cov
        # now fill the diagonal with the variance:
        return result

    def _rotation_matrixes(self, cov):
        """
        Calculates the rotation matrixes E_* and D_* for the given
        covariance matrix according to Argamon
        """
        ev, E = linalg.eig(cov)

        # Only use eigenvalues != 0 and the corresponding eigenvectors
        select = np.array(ev, dtype=bool)
        D_ = np.diag(ev[select])
        E_ = E[:, select]
        return (E_, D_)

    def _rotated_delta(self, E, Dinv, corpus):
        """Performs the actual delta calculation."""
        deltas = pd.DataFrame(index=corpus.columns, columns=corpus.columns)
        nwords = corpus.index.size
        for d1, d2 in itertools.combinations(corpus.columns, 2):
            diff = (corpus.loc[:, d1] - corpus.loc[:, d2]).reshape(nwords, 1)
            delta = diff.T.dot(E).dot(Dinv).dot(E.T).dot(diff)
            #       ------     -      ----      ---      ----
            # dim:   1,n      n,m     m,m       m,n       n,1  -> 1,1
            deltas.at[d1, d2] = delta[0, 0]
            deltas.at[d2, d1] = delta[0, 0]
        return deltas.fillna(0)

    def rotated_delta(self, corpus, refcorpus, cov_alg='nonbiased'):
        """
        Calculates $\Delta_{Q,\not\perp}^{(n)}$ according to Argamon, i.e.
        the axis-rotated quadratic delta using eigenvalue decomposition to
        rotate the feature space according to the word frequency covariance
        matrix calculated from a reference corpus

        :param corpus: Pandas Dataframe (word×documents -> word frequencies) for
                       which to calculate the document deltas
        :param refcorpus: Pandas Dataframe with the reference corpus
        :cov_alg: covariance algorithm choice, 'argamon', 'nonbiased' or a function
        """
        if refcorpus is None:
            raise Exception("rotated delta requires a reference corpus.")
        refc = refcorpus.loc[corpus.index].fillna(0)
        if callable(cov_alg):
            cov = cov_alg(refc)
        elif cov_alg == 'argamon':
            cov = self._cov_matrix(refc)
        elif cov_alg == 'nonbiased':
            cov = refc.T.cov()
        E_, D_ = self._rotation_matrixes(cov)
        D_inv = linalg.inv(D_)
        return self._rotated_delta(E_, D_inv, corpus)

    def get_linkage(self, stat_linkage_method):
        #create the datamodel which is needed as input for the dendrogram
        #only method ward demands a redundant distance matrix while the others seem to get different
        #results with a redundant matrix and with a flat one, latter seems to be ok.
        #see https://github.com/scipy/scipy/issues/2614  (not sure this is still an issue)
        if stat_linkage_method == "ward":
            z = sch.linkage(self.results, method='ward', metric='euclidean')
        else:
            #creating a flat representation of the dist matrix
            deltas_flat = ssd.squareform(self.results)
            z = sch.linkage(deltas_flat, method=stat_linkage_method, metric='euclidean')
        return z


class Figure():
    z = None
    titles = []
    fig_orientation = ""
    figure_font_size = 0
    figure_title = ""
    stat_mfwords = 0
    delta_algorithm = {}
    delta_choice = 0
    save_sep = ""
    figure_show = False

    def __init__(self, z, titles, fig_orientation,
                 figure_font_size, figure_title, stat_mfwords, delta_algorithm,
                 delta_choice, save_sep, figure_show):
        """
        creates a dendogram which is displayed and saved to a file
        there is a bug (I think) in the dendrogram method in scipy.cluster.hierarchy (probably)
        or in matplotlit. If you choose  orientation="left" or "bottom" the method get_text on the labels
        returns an empty string.
        """
        self.z = z
        self.titles = titles
        self.fig_orientation = fig_orientation
        self.figure_font_size = figure_font_size
        self.figure_title = figure_title
        self.stat_mfwords = stat_mfwords
        self.delta_algorithm = delta_algorithm
        self.delta_choice = delta_choice
        self.save_sep = save_sep
        self.figure_show = figure_show

    def show(self):
        #clear the figure
        plt.clf()

        if self.fig_orientation == "top":
            rotation = 90
        elif self.fig_orientation == "left":
            rotation = 0
        dendro_data = sch.dendrogram(self.z, orientation=self.fig_orientation, labels=self._shorten_labels(self.titles),
                                     leaf_rotation=rotation, link_color_func=lambda k: 'k',
                                     leaf_font_size=self.figure_font_size)
        #get the axis
        ax = plt.gca()
        self._color_coding_author_names(ax, self.fig_orientation)
        plt.title(self.figure_title)
        plt.xlabel(str(self.stat_mfwords) + " most frequent words. " + self.delta_algorithm[self.delta_choice])
        plt.tight_layout(2)
        if self.figure_show:
            plt.show()
        return dendro_data, plt

    def save(self, plt):
        plt.savefig(self._result_filename(self.figure_title, self.save_sep, self.stat_mfwords,
                                          self.delta_algorithm, self.delta_choice) + ".png")

    def save_results(self, mfw_corpus, deltas, figure_title, save_sep, stat_mfwords, delta_algorithm, delta_choice):
        """saves results to files
        :param mfw_corpus: the DataFrame shortened to the most frequent words
        :param deltas: a DataFrame containing the Delta distance between the texts
        """
        mfw_corpus.to_csv("mfw_corpus.csv", encoding="utf-8")
        deltas.to_csv(self._result_filename(figure_title, save_sep, stat_mfwords, delta_algorithm, delta_choice)
                      + ".results.csv", encoding="utf-8")


    def _get_author_surname(self, author_complete):
        """extract surname from complete name
        :param author_complete:
        :rtype : str
        """
        return author_complete.split(",")[0]

    def _shorten_title(self, title):
        """shortens title to a meaningful but short string
        :param title:
        :rtype : str
        """
        junk = ["Ein", "Eine", "Der", "Die", "Das"]
        title_parts = title.split(" ")
        #getting rid of file ending .txt
        if ".txt" in title_parts[-1]:
            title_parts[-1] = title_parts[-1].split(".")[0]
        #getting rid of junk at the beginning of the title
        if title_parts[0] in junk:
            title_parts.remove(title_parts[0])
        t = " ".join(title_parts)
        if len(t) > 25:
            return t[0:24]
        else:
            return t

    def _shorten_label(self, label):
        """shortens one label consisting of authorname and title
        :param label: a string containg the long version
        :rtype: str
        """
        if "__" in label:
            label = label.replace("__", "_")
        author, title = label.split("_")
        return self._get_author_surname(author) + "_" + self._shorten_title(title)

    def _shorten_labels(self, labels):
        """
        shortens author and titles of novels in a useful way
        open problem: similar titles which start with the same noun
        :param labels: list of file names using author_title
        :rtype: str
        """
        new_labels = []
        for l in labels:
            new_labels.append(self._shorten_label(l))
        return new_labels

    def _color_coding_author_names(self, ax, fig_orientation):
        """color codes author names
        :param ax: a matplotlib axis as created by the dendogram method
        """
        lbls = []
        #get the labels from axis
        if fig_orientation == "left":
            lbls = ax.get_ymajorticklabels()
        elif fig_orientation == "top":
            lbls = ax.get_xmajorticklabels()
        colors = ["r", "g", "b", "m", "k", "Olive", "SaddleBrown", "CadetBlue", "DarkGreen", "Brown"]
        cnt = 0
        authors = {}
        new_labels = []
        for lbl in lbls:
            author, title = lbl.get_text().split("_")
            if author in authors:
                lbl.set_color(authors[author])
            else:
                color = colors[cnt]
                authors[author] = color
                lbl.set_color(color)
                cnt += 1
                if cnt == 9:
                    cnt = 0
            lbl.set_text(author + " " + title)
            new_labels.append(author + " " + title)
        if fig_orientation == "left":
            ax.set_yticklabels(new_labels)
        elif fig_orientation == "top":
            ax.set_xticklabels(new_labels)

    def _format_time(self, save_sep):
        """
        helper method. date and time are used to create a string for inclusion into the filename
        rtype: str
        """
        dt = datetime.now()
        return u'{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(str(dt.year), save_sep, str(dt.month),
                                                     save_sep, str(dt.day), save_sep,
                                                     str(dt.hour), save_sep, str(dt.minute))

    def _result_filename(self, figure_title, save_sep, stat_mfwords, delta_algorithm, delta_choice):
        """
        helper method to format the filename for the image which is saved to file
        the name contains the title and some info on the chosen statistics
        """
        return figure_title + save_sep + str(stat_mfwords) \
               + " mfw. " + delta_algorithm[delta_choice] \
               + save_sep + self._format_time(save_sep)


class Config():
    cfg = None

    def __init__(self):
        config = profig.Config("pydelta.ini")
        self.cfg = config
        self.initialize()
        self.get_commandline()

    def initialize(self):
        """
        writes/reads a default configuration to pydelta.ini. If you want to change these parameters, use the ini file
        """
        self.cfg.init("files.ini", False, comment="if true writes a configuration file to disk")
        self.cfg.init("files.subdir", "corpus", comment="the subdirectory containing the text files used as input")
        self.cfg.init("files.refcorpus", "refcorpus", comment="the reference corpus required for some methods")
        self.cfg.init("files.encoding", "utf-8", comment="the file encoding for the input files")
        self.cfg.init("files.use_wordlist", False, comment="not implemented yet")
        self.cfg.init("data.use_corpus", False, comment="use a corpus of word frequencies from file; filename is " +
                                                        "corpus.csv")
        self.cfg.init("data.limit", 200000, comment="amount of file to be processed in chars. only useful in " +
                                                    "combination with set_limit")
        self.cfg.init("data.set_limit", False, comment="only use a specified amount of the file. NOT TESTED")
        self.cfg.init("data.lower_case", False, comment="convert all words to lower case")
        #self.cfg.init("save.fileout", "corpus_words.csv", comment="sets the file name " +
        #                                                        "to save the corpus words into a file")
        self.cfg.init("save.complete_corpus", False, comment="save the complete word + freq lists")
        self.cfg.init("save.save_results", True, comment="the results (i.e. the delta measures) are written " +
                                                         "into the file results.csv. The mfwords are saved too.")
        self.cfg.init("save.sep", "-", comment="sperates time and statistic information in the image filename")
        self.cfg.init("stat.mfwords", 2000, comment="number of most frequent words to use " +
                                                    "in the calculation of delta. 0 for all words")
        self.cfg.init("stat.delta_algorithm", ["Classic Delta",  # XXX
                                               "Argamon's linear Delta",
                                               "Argamon's quadratic Delta",
                                               "Argamon's axis-rotated Delta"], comment="available delta algorithms")
        self.cfg.init("stat.delta_choice", 0, comment="choice of the delta algorithm. 0 = classic, 1 = linear usw.")
        self.cfg.init("stat.linkage_method", "ward", comment="method how the distance between the newly formed " +
                                                             "cluster and each candidate is calculated. Valid " +
                                                             "values: 'ward', 'single', 'average', 'complete',   " +
                                                             "'weighted'. See documentation on " +
                                                             "scipy.cluster.hierarchy.linkage for details.")
        self.cfg.init("stat.evaluate", False, comment="evaluation of the results. Only useful if there are always" +
                                                      "more than 2 texts by an author and the attribution of all " +
                                                      "texts is known.")
        self.cfg.init("figure.title", 'Stylistic analysis using Delta', comment="title is used in the " +
                                                                                "figure and the name of the " +
                                                                                "file containing the figure")
        self.cfg.init("figure.filenames_labels", True, comment="use the filename of the texts as labels")
        self.cfg.init("figure.fig_orientation", "left", comment="orientation of the figure. allowed values: " +
                                                                "'left', (author and title at the left side)" +
                                                                "'top' (author/title at the bottom)")
        self.cfg.init("figure.font_size", 11, comment="font-size for labels in the figure")
        self.cfg.init("figure.show", True, comment="show figure interactively if true")

        #writes configuration files if it doesn't exist, otherwise reads in values from there
        self.cfg.sync()


    def get_commandline(self):
        """
        allows user to override all settings using commandline arguments.
        """
        help_msg = " ".join([str(x) for x in self.cfg])
        parser = argparse.ArgumentParser()
        parser.add_argument('-O', dest='options', action='append',
                            metavar='<key>:<value>', help='Overrides an option in the config file.' +
                                                          '\navailable options:\n' + help_msg)
        args = parser.parse_args()
        # update option values
        if args.options is not None:
            self.cfg.update(opt.split(':') for opt in args.options)


def check_max(s):
    max_value = 0
    aname = s.name.split("_")[0]
    for i in s.index:
        name = i.split("_")[0]
        if name == aname:
            if s[i] > max_value:
                max_value = s[i]
    return max_value


def classified_correctly(s, max_value):
    """
    ATT: DON'T USE THIS. checks whether the distance of a given text to texts of other authors is smaller than to
     texts of the same author
    :param: s: a pd.Series containing deltas
    :max_value: the largest distance to a text of the same author
    """
    #if only one text of an author is in the set, an evaluation of the  clustering makes no sense
    if max_value == 0:
        return None
    aname = s.name.split("_")[0]
    for i in s.index:
        name = i.split("_")[0]
        if name != aname:
            if s[i] < max_value:
                return False
    return True


def error_eval(l):
    """
    trival check of a list of numbers for 'errors'. An error is defined as i - (i-1)  != 1
    :param: l: a list of numbers representing the position of the author names in the figure labels
    :rtype: int
    """
    errors = 0
    d = None
    for i in range(len(l)):
        if d is None:
            d = 0
        else:
            if l[i] - l[i - 1] > 1:
                errors += 1
    return errors


def evaluate_results(fig_data):
    """
    evaluates the results on the basis of the dendrogram
    :param: fig_data: representation of the dendrogram as returned by
                      scipy.cluster.hierarchy.dendrogram
    """
    ivl = fig_data['ivl']
    authors = {}
    for i in range(len(ivl)):
        au = ivl[i].split("_")[0]
        if au not in authors:
            authors[au] = [i]
        else:
            authors[au].append(i)

    errors = 0
    for x in authors.keys():
        errors += error_eval(authors[x])

    print("attributions - errors: ", len(ivl), " - ", errors)


def _purify_delta(delta):
    """
    Retains only the non-duplicate meaningful deltas.

    I.e. the diagonal and the lower left triangle are set to np.na.
    """
    return delta.where(np.triu(np.ones(delta.shape), k=1))


def delta_values(delta):
    """
    Converts the given n×n Delta matrix to a nC2 long series of distinct delta
    values – i.e. duplicates from the lower triangle and zeros from the
    diagonal are removed.
    """
    return _purify_delta(delta).unstack().dropna()


def normalize_delta(delta):
    """Normalizes the given delta matrix using the z-Score."""
    deltas = delta_values(delta)
    return (delta - deltas.mean()) / deltas.std()


def _author(filename):
    return filename.partition("_")[0]


def _partition_deltas(deltas, indexfunc=_author):
    """
    Partitions the given deltas by same author.
    """
    same = pd.DataFrame(index=deltas.index, columns=deltas.index)
    diff = pd.DataFrame(index=deltas.index, columns=deltas.index)
    # c'mon. This must go more elegant?
    for d1, d2 in itertools.combinations(deltas.columns, 2):
        if indexfunc(d1) == indexfunc(d2):
            same.at[d1, d2] = deltas.at[d1, d2]
        else:
            diff.at[d1, d2] = deltas.at[d1, d2]
    return (same, diff)


def evaluate_deltas(deltas, verbose=True):
    """
    Simple delta quality score for the given delta matrix:
    The difference between the means of the standardized differences between
    works of different authors and works of the same author.

    Larger scores = better.

    :param verbose: (default True) also print the score and intermediate
    results
    """
    d_equal, d_different = _partition_deltas(normalize_delta(deltas))
    equal, different = delta_values(d_equal), delta_values(d_different)
    score = different.mean() - equal.mean()
    if verbose:
        print("Normalized deltas for same author, mean=%g, std=%g:" %
              (equal.mean(), equal.std()))
        print("Normalized deltas for different author, mean=%g, std=%g:" %
              (different.mean(), different.std()))
        print("### Simple Delta Quality Score = %g" % score)
    return score


def main():
    #read configuration
    cfg = Config()
    #uses existing corpus or processes a new set of files
    if cfg.cfg["data.use_corpus"]:
        print("reading corpus from file corpus.csv")
        corpus = Corpus()
    else:
        corpus = Corpus(cfg.cfg['files.subdir'], None, cfg.cfg["files.encoding"], cfg.cfg["data.limit"],
                        cfg.cfg["data.set_limit"], cfg.cfg["data.lower_case"])
        if cfg.cfg["save.complete_corpus"]:
            corpus.save()

    #creates a smaller table containing just the mfwords
    mfw_corpus = corpus.get_mfw_table(cfg.cfg['stat.mfwords'])
    mfw_corpus.save()
    #calculates the specified delta
    delta_choice = cfg.cfg["stat.delta_choice"]
    #create reference corpus for Argamon's axis rotated Delta
    if delta_choice == 3:
        refcorpus = Corpus(subdir=cfg.cfg['files.refcorpus'], encoding=cfg.cfg["files.encoding"])
    else:
        refcorpus = None
    #calculate Delta
    deltas = Delta(mfw_corpus, delta_choice, refcorpus)
    #setup the figure using the deltas
    fig = Figure(deltas.get_linkage(cfg.cfg["stat.linkage_method"]), deltas.results.index,
                 cfg.cfg["figure.fig_orientation"], cfg.cfg["figure.font_size"], cfg.cfg["figure.title"],
                 cfg.cfg["stat.mfwords"], cfg.cfg["stat.delta_algorithm"], cfg.cfg["stat.delta_choice"],
                 cfg.cfg["save.sep"], cfg.cfg["figure.show"])
    #create the figure
    dendro_dat, plot = fig.show()
    fig.save(plot)
    #prints results from evaluation
    if cfg.cfg["stat.evaluate"]:
        evaluate_results(dendro_dat)
        evaluate_deltas(deltas)


if __name__ == '__main__':
    main()

