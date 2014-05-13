#!/usr/bin/env python3 
"""
calculates Burrow's Delta and Argamon's proposed variations
tbd:
- add a more solid evaluation for the results.
- rotated_delta: convert the complex results, use abs()?
- load deltas from file (to check our results against stylo)
- set delta alg., mfwords and linkage alg from command line
- replace print statements with logging mechanism?
- refactor in a more OO way to improve reuse
- write a gui to set all the configuration information, is this necessary?

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


def get_configuration():
    """
    writes/reads a default configuration to pydelta.ini. If you want to change these parameters, use the ini file
    """
    config = profig.Config("pydelta.ini")
    config.init("files.ini", False, comment="if true writes a configuration file to disk")
    config.init("files.subdir", "corpus", comment="the subdirectory containing the text files used as input")
    config.init("files.refcorpus", "refcorpus", comment="the reference corpus required for some methods")
    config.init("files.encoding", "utf-8", comment="the file encoding for the input files")
    config.init("files.use_wordlist", False, comment="not implemented yet")
    config.init("data.use_corpus", False, comment="use a corpus of word frequencies from file; filename is " +
                                                  "corpus.csv")
    config.init("data.limit", 200000, comment="amount of file to be processed in chars. only useful in " +
                                              "combination with set_limit")
    config.init("data.set_limit", False, comment="only use a specified amount of the file. NOT TESTED")
    config.init("data.lower_case", False, comment="convert all words to lower case")
    #config.init("save.fileout", "corpus_words.csv", comment="sets the file name " +
    #                                                        "to save the corpus words into a file")
    config.init("save.complete_corpus", False, comment="save the complete word + freq lists")
    config.init("save.save_results", True, comment="the results (i.e. the delta measures) are written " +
                                                   "into the file results.csv. The mfwords are saved too.")
    config.init("save.sep", "-", comment="sperates time and statistic information in the image filename")
    config.init("stat.mfwords", 2000, comment="number of most frequent words to use " +
                                                    "in the calculation of delta. 0 for all words")
    config.init("stat.delta_algorithm", ["Classic Delta", # XXX
                                               "Argamon's linear Delta",
                                               "Argamon's quadratic Delta",
                                               "Argamon's axis-rotated Delta"], comment="available delta algorithms")
    config.init("stat.delta_choice", 0, comment="choice of the delta algorithm. 0 = classic, 1 = linear usw.")
    config.init("stat.linkage_method", "ward", comment="method how the distance between the newly formed " +
                                                             "cluster and each candidate is calculated. Valid " +
                                                             "values: 'ward', 'single', 'average', 'complete',   " +
                                                             "'weighted'. See documentation on " +
                                                             "scipy.cluster.hierarchy.linkage for details.")
    config.init("stat.evaluate", False, comment="evaluation of the results. Only useful if there are always" +
                                                      "more than 2 texts by an author and the attribution of all " +
                                                      "texts is known.")
    config.init("figure.title", 'Stylistic analysis using Delta', comment="title is used in the " +
                                                                              "figure and the name of the " +
                                                                              "file containing the figure")
    config.init("figure.filenames_labels", True, comment="use the filename of the texts as labels")
    config.init("figure.fig_orientation", "left", comment="orientation of the figure. allowed values: " +
                                                          "'left', (author and title at the left side)" +
                                                          "'top' (author/title at the bottom)")
    config.init("figure.font_size", 11, comment="font-size for labels in the figure")

    #writes configuration files if it doesn't exist, otherwise reads in values from there
    config.sync()

    return config


def get_commandline(config):
    """
    allows user to override all settings using commandline arguments.
    """
    help_msg = " ".join([str(x) for x in config])
    parser = argparse.ArgumentParser()
    parser.add_argument('-O', dest='options', action='append',
                        metavar='<key>:<value>', help='Overrides an option in the config file.'+
                                                      '\navailable options:\n' + help_msg)
    args = parser.parse_args()
    # update option values
    if args.options is not None:
        config.update(opt.split(':') for opt in args.options)
    return config



def process_files(subdir, encoding="utf-8", limit=2000, set_limit=False, lower_case=False):
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
        print("The directory " + subdir + " doesn't exist. \nPlease add a directory " +
              "with text files.\n")
        sys.exit(1)
    filelist = glob.glob(subdir + os.sep + "*.txt")
    list_of_wordlists = []
    for file in filelist:
        list_of_wordlists.append(tokenize_file(file, encoding, limit, set_limit, lower_case))
    corpus_words = pd.DataFrame(list_of_wordlists).fillna(0).T
    return corpus_words


def tokenize_file(filename, encoding, limit, set_limit, lower_case):
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


def save_file(corpus_words, config):
    """
    saves wordlists to file
    """
    print("Saving wordlist to file " + config["save.fileout"])
    corpus_words.to_csv("corpus_words.csv", encoding="utf-8", na_rep=0, quoting=csv.QUOTE_NONNUMERIC)


def preprocess_mfw_table(corpus, mfwords):
    """
    sorts the table containing the frequency lists
    by the sum of all word freq
    returns the corpus list shortened to the most frequent words
    number defined by mfwords
    """
    #nifty trick to get it sorted according to sum
    #not from me :-)
    new_corpus = corpus.loc[(-corpus.sum(axis=1)).argsort()]
    #slice only mfwords from total list
    if mfwords > 0:
        return new_corpus[:mfwords]
    else:
        return new_corpus


def corpus_stds(corpus):
    """calculates std for all words of the corpus
       returns a pd.Series containing the means and the
       words as index"""
    return corpus.std(axis=1)


def corpus_medians(corpus):
    """calculates medians for all words of the corpus
       returns a pd.Series containing the medians and the
       words as index"""
    return corpus.median(axis=1)


def diversity(values):
    """
    calculates the spread or diversity (wikipedia) of a laplace distribution of values
    see Argamon's Interpreting Burrow's Delta p. 137 and
    http://en.wikipedia.org/wiki/Laplace_distribution
    couldn't find a ready-made solution in the python libraries
    :param values: a pd.Series of values
    """
    return (values - values.median()).abs().sum() / values.size


def corpus_diversities(corpus):
    """
    calculate the 'spread' of word distributions assuming they are laplace dist.
    """
    return corpus.apply(diversity, axis=1)


#not used at the moment
def corpus_means(corpus):
    """calculates the means for all words of the corpus
       returns a pd.Series containing the means and the
       words as index"""
    return corpus.mean(axis=1)


def calculate_delta(corpus, delta_choice, refcorpus=None):
    """
    chooses the algorithm for the calculation of delta
    after rewriting classic_delta this can be moved there
    """
    # XXX use functions directly & set a title attribute on them?
    if delta_choice == 0:
        return classic_delta(corpus)
    elif delta_choice == 1:
        return linear_delta(corpus)
    elif delta_choice == 2:
        return quadratic_delta(corpus)
    elif delta_choice == 3:
        return rotated_delta(corpus, refcorpus)
    else:
        #tbd: use raise Exception for the following
        print("ERROR: You have to choose an algorithm for Delta.")
        sys.exit(1)


def classic_delta(corpus):
    """
    calculates Delta in the simplified form proposed by Argamon
    """
    #print("using classic delta")
    stds = corpus_stds(corpus)
    deltas = pd.DataFrame(index=corpus.columns, columns=corpus.columns)
    for i, j in itertools.combinations(corpus.columns, 2):
        delta = ((corpus[i] - corpus[j]).abs() / stds).sum() / len(corpus.index)
        deltas.at[i, j] = delta
        deltas.at[j, i] = delta
    return deltas.fillna(0)


def quadratic_delta(corpus):
    """
    Argamon's quadratic Delta
    """
    #print("using quadratic delta")
    vars_ = corpus_stds(corpus) ** 2
    deltas = pd.DataFrame(index=corpus.columns, columns=corpus.columns)
    for i, j in itertools.combinations(corpus.columns, 2):
        delta = ((corpus[i] - corpus[j]) ** 2 / vars_).sum()
        deltas.at[i, j] = delta
        deltas.at[j, i] = delta
    return deltas.fillna(0)


def linear_delta(corpus):
    """
    Argamon's linear Delta
    """
    #print("using linear delta")
    diversities = corpus_diversities(corpus)
    deltas = pd.DataFrame(index=corpus.columns, columns=corpus.columns)

    for i, j in itertools.combinations(corpus.columns, 2):
        delta = ((corpus[i] - corpus[j]).abs() / diversities).sum()
        deltas.at[i, j] = delta
        deltas.at[j, i] = delta
    return deltas.fillna(0)


# rotated quadratic delta. This could be moved into the main delta function
# when it is finished

def _cov_matrix(corpus):
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
        result.at[w,w] = (dev.loc[w]**2).sum() / documents

    # FIXME hier ist noch Optimierungspotential, 
    # bei 2000 Wörtern läuft das ewig. 
    # Ggf. corpus.loc[w]-means.at[w] cachen? ob's das bringt?
    for i, j in itertools.combinations(corpus.index, 2):
            cov = (dev.loc[i] * dev.loc[j]).sum() / documents
            result.at[i,j] = cov
            result.at[j,i] = cov
    # now fill the diagonal with the variance:
    return result


def _rotation_matrixes(cov):
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


def _rotated_delta(E, Dinv, corpus):
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

def rotated_delta(corpus, refcorpus, cov_alg='argamon'):
    r"""
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
        cov = _cov_matrix(refc)
    elif cov_alg == 'nonbiased':
        cov = refc.cov()
    E_, D_ = _rotation_matrixes(cov)
    D_inv = linalg.inv(D_)
    return _rotated_delta(E_, D_inv, corpus)

def get_author_surname(author_complete):
    """extract surname from complete name
    :param author_complete:
    :rtype : str
    """
    return author_complete.split(",")[0]


def shorten_title(title):
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


def shorten_label(label):
    """shortens one label consisting of authorname and title
    :param label: a string containg the long version
    :rtype: str
    """
    if "__" in label:
        label = label.replace("__", "_")
    author, title = label.split("_")
    return get_author_surname(author) + "_" + shorten_title(title)


def shorten_labels(labels):
    """
    shortens author and titles of novels in a useful way
    open problem: similar titles which start with the same noun
    :param labels: list of file names using author_title
    :rtype: str
    """
    new_labels = []
    for l in labels:
        new_labels.append(shorten_label(l))
    return new_labels


def color_coding_author_names(ax, fig_orientation):
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


def display_results(deltas, stat_linkage_method, fig_orientation,
        figure_font_size, figure_title, stat_mfwords, delta_algorithm,
        delta_choice, save_sep):
    """
    creates a dendogram which is displayed and saved to a file
    there is a bug (I think) in the dendrogram method in scipy.cluster.hierarchy (probably)
    or in matplotlit. If you choose  orientation="left" or "bottom" the method get_text on the labels
    returns an empty string.
    """
    #clear the figure
    plt.clf()
    #create the datamodel which is needed as input for the dendrogram
    #only method ward demands a redundant distance matrix while the others seem to get different
    #results with a redundant matrix and with a flat one, latter seems to be ok.
    #see https://github.com/scipy/scipy/issues/2614  (not sure this is still an issue)
    if stat_linkage_method == "ward":
        z = sch.linkage(deltas, method='ward', metric='euclidean')
    else:
        #creating a flat representation of the dist matrix
        deltas_flat = ssd.squareform(deltas)
        z = sch.linkage(deltas_flat, method=stat_linkage_method, metric='euclidean')
    #print("linkage method: ", config["stat.linkage_method"])
    #create the dendrogram
    if fig_orientation == "top":
        rotation = 90
    elif fig_orientation == "left":
        rotation = 0
    dendro_data = sch.dendrogram(z, orientation=fig_orientation, labels=shorten_labels(deltas.index),
                                 leaf_rotation=rotation, link_color_func=lambda k: 'k',
                                 leaf_font_size=figure_font_size)
    #get the axis
    ax = plt.gca()
    color_coding_author_names(ax, fig_orientation)
    plt.title(figure_title)
    plt.xlabel(str(stat_mfwords) + " most frequent words. " + delta_algorithm[delta_choice])
    plt.tight_layout(2)
    plt.savefig(result_filename(figure_title, save_sep, stat_mfwords, delta_algorithm, delta_choice) + ".png")
    plt.show()
    return dendro_data


def format_time(save_sep):
    """
    helper method. date and time are used to create a string for inclusion into the filename
    rtype: str
    """
    dt = datetime.now()
    return u'{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(str(dt.year), save_sep, str(dt.month),
                                                 save_sep, str(dt.day), save_sep,
                                                 str(dt.hour), save_sep, str(dt.minute))


def result_filename(figure_title, save_sep, stat_mfwords, delta_algorithm, delta_choice):
    """
    helper method to format the filename for the image which is saved to file
    the name contains the title and some info on the chosen statistics
    """
    return figure_title + save_sep + str(stat_mfwords) \
           + " mfw. " + delta_algorithm[delta_choice] \
           + save_sep + format_time(save_sep)


def save_results(mfw_corpus, deltas, figure_title, save_sep, stat_mfwords, delta_algorithm, delta_choice):
    """saves results to files
    :param mfw_corpus: the DataFrame shortened to the most frequent words
    :param deltas: a DataFrame containing the Delta distance between the texts
    """
    mfw_corpus.to_csv("mfw_corpus.csv", encoding="utf-8")
    deltas.to_csv(result_filename(figure_title, save_sep, stat_mfwords, delta_algorithm, delta_choice)
                                  + ".results.csv", encoding="utf-8")


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


def main():
    #reads pydelta.ini or uses defaults
    config = get_configuration()
    #reads overriding options from commandline
    config = get_commandline(config)

    #only write pydelta.ini and then exit
    if config["files.ini"]:
        sys.exit()

    #uses existing corpus or processes a new set of files
    if config["data.use_corpus"]:
        print("reading corpus from file corpus.csv")
        corpus = pd.read_csv("corpus.csv", index_col=0)
    else:
        corpus = process_files(config['files.subdir'], config["files.encoding"], config["data.limit"], config["data.set_limit"], config["data.lower_case"])
        if config["save.complete_corpus"]:
            corpus.to_csv("corpus.csv", encoding="utf-8")

    #creates a smaller table containing just the mfwords
    mfw_corpus = preprocess_mfw_table(corpus, config["stat.mfwords"])
    #calculates the specified delta

    delta_choice = config["stat.delta_choice"]
    if delta_choice == 3:
        refcorpus = process_files(config['files.refcorpus'], config["files.encoding"], config["data.limit"],
                               config["data.set_limit"], config["data.lower_case"])
    else:
        refcorpus = None
    deltas = calculate_delta(mfw_corpus, delta_choice, refcorpus)
    #creates a clustering using linkage and then displays the dendrogram
    fig = display_results(deltas, config["stat.linkage_method"], config["figure.fig_orientation"],
                          config["figure.font_size"], config["figure.title"], config["stat.mfwords"],
                          config["stat.delta_algorithm"], config["stat.delta_choice"], config["save.sep"])
    save_results(mfw_corpus, deltas, config["figure.title"], config["save.sep"], config["stat.mfwords"],
                 config["stat.delta_algorithm"], config["stat.delta_choice"])
    #prints results from evaluation
    if config["stat.evaluate"]:
        evaluate_results(fig)


if __name__ == '__main__':
    main()
