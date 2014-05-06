#!/usr/bin/env python3 
"""
calculates Burrow's Delta and - hopefully - some variants of it
tbd:
- improve timing for building the tables
- improve the output figure
- write a handler for all the general variables, allowing to use a configuration file
  and a gui to set them
"""
import sys
import glob
import re
import collections
import os
import pandas as pd
import numpy as np
import numpy.linalg as npl
import csv
import scipy.cluster.hierarchy as sch
import matplotlib.pylab as plt
import itertools
from datetime import datetime
import cProfile

#number of words to use from the wordlist
#set to 0 to use all
mfwords = 2000

#use existing wordlist
use_wordlist = False

#where to find the corpus
subdir = "corpus3"

#file_encoding
encoding = "utf-8"

#switch to extract only a sample. NOT tested yet
set_limit = False

#sample size
limit = 200000

#all words in lower case?
lower_case = False

#name of the file where the complete wordlist is saved
fileout = "corpus_words.csv"

#use filenames as labels in plot
filenames_labels = True

#save results to file results.csv
save_results = True

#separates information in filename
sep = "-"

#title in figure and filename
title = 'Dramatiker um 1800'

#name of the chosen algorithm
delta_algorithm = {1: "Classic Delta",
                   2: "Argamon's linear Delta",
                   3: "Argamon's quadratic Delta"}

delta_choice = delta_algorithm[1]


def process_files(encoding="utf-8"):
    """
    preprocessing all files ending with *.txt in corpus subdir
    all files are tokenized
    a table of all word and their freq in all texts is created
    format
            filename1 filename2 filename3
    word      nr        nr
    word      nr        nr
    """
    filelist = glob.glob(subdir + os.sep + "*.txt")
    list_of_wordlists = []
    for file in filelist:
        list_of_wordlists.append(tokenize_file(file, encoding))
    corpus_words = pd.DataFrame(list_of_wordlists).fillna(0)
    return corpus_words.T


def tokenize_file(filename, encoding):
    """
    tokenizes file and returns an unordered pandas.DataFrame
    containing the words and frequencies
    standard encoding = utf-8
    """
    all_words = collections.defaultdict(int)

    #read file, tokenize it, count words
    read_text_length = 0
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


def save_file(corpus_words):
    """
    saves wordlists to file
    using global var save_file
    """
    print("Saving wordlist to file " + fileout)
    corpus_words.to_csv("corpus_words.csv", encoding="utf-8", na_rep=0, quoting=csv.QUOTE_NONNUMERIC)


def preprocess_mfw_table(corpus, mfwords=mfwords):
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
    if mfwords != 0:
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
    return corpus.apply(diversity, axis=1)


#not used at the moment
def corpus_means(corpus):
    """calculates the means for all words of the corpus
       returns a pd.Series containing the means and the
       words as index"""
    return corpus.mean(axis=1)


def calculate_delta(corpus):
    """
    chooses the algorithm for the calculation of delta
    after rewriting classic_delta this can be moved there
    """
    if delta_choice == delta_algorithm[1]:
        return classic_delta1(corpus)
    elif delta_choice == delta_algorithm[2]:
        return linear_delta(corpus)
    elif delta_choice == delta_algorithm[3]:
        return quadratic_delta(corpus)
    else:
        #tbd: use raise Exception for the following
        print("ERROR: You have to choose an algorithm for Delta.")
        sys.exit(1)


def classic_delta(corpus):
    stds = corpus_stds(corpus)
    deltas = pd.DataFrame(index=corpus.columns, columns=corpus.columns)
    for j in range(len(corpus.columns)):
        for h in range(j, len(corpus.columns)):
            if j != h:
                delta = sum(abs(corpus[corpus.columns[j]] - corpus[corpus.columns[h]]) / stds) / mfwords
                deltas[corpus.columns[j]][corpus.columns[h]] = delta
                deltas[corpus.columns[h]][corpus.columns[j]] = delta
            else:
                deltas[corpus.columns[h]][corpus.columns[j]] = 0
    return deltas


def classic_delta1(corpus):
    """
    calculates Delta in the simplified form proposed by Argamon
    """
    stds = corpus_stds(corpus)
    mfwords = c.index.size
    deltas = pd.DataFrame(index=corpus.columns, columns=corpus.columns)
    for i, j in itertools.combinations(corpus.columns, 2):
        delta = ((corpus[i] - corpus[j]).abs() / stds).sum() / mfwords
        deltas.at[i, j] = delta
        deltas.at[j, i] = delta
    return deltas.fillna(0)

def quadratic_delta(corpus):
    """
    Argamon's quadratic Delta
    """
    print ("using quadratic delta")
    vars_ = corpus_stds(corpus)**2
    mfwords = c.index.size
    deltas = pd.DataFrame(index=corpus.columns, columns=corpus.columns)
    for i, j in itertools.combinations(corpus.columns, 2):
        delta = ((corpus[i]-corpus[j])**2 / vars_).sum()
        deltas.at[i, j] = delta
        deltas.at[j, i] = delta
    return deltas.fillna(0)



def linear_delta(corpus):
    """
    Argamon's linear Delta
    """
    print ("using linear delta")
    diversities = corpus_diversities(corpus)
    deltas = pd.DataFrame(index=corpus.columns, columns=corpus.columns)

    for i, j in itertools.combinations(corpus.columns, 2):
        delta = ((corpus[i] - corpus[j]).abs() / diversities).sum()
        deltas.at[i, j] = delta
        deltas.at[j, i] = delta
    return deltas.fillna(0)

def cov_matrix(corpus):
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

def rotation_matrixes(cov):
    """
    Calculates the rotation matrixes E_* and D_* for the given
    covariance matrix according to Argamon
    """
    ev, E = npl.eig(cov)
    D = np.diag(ev)
    # lustigerweise hab ich in meinen experimenten _nie_ ein eigvals_i=0
    # gefunden. D.h. die Reduktion können wir uns sparen:
    if 0 in ev:
        raise Exception("Oops. Seems we need to implement the reduction function.")
    return (E, D)


def delta_rotated(corpus, cov):
    """
    Calculates $\Delta_{Q,\not\perp}^{(n)}$ according to Argamon, i.e.
    the axis-rotated quadratic delta using eigenvalue decomposition to 
    rotate the feature space according to the word frequency covariance 
    matrix calculated from a reference corpus
    """
    E, D = rotation_matrixes(cov)
    # XXX



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


def format_time():
    """
    helper method. date and time are used to create a string for inclusion into the filename
    rtype: str
    """
    dt = datetime.now()
    return u'{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(str(dt.year), sep, str(dt.month), sep, str(dt.day), sep,
                                                     str(dt.hour), sep, str(dt.minute))


def color_coding_author_names(ax):
    """color codes author names
    :param ax: a matplotlib axis as created by the dendogram method
    """
    #get the labels from the x - axis
    ylbls = ax.get_ymajorticklabels()
    colors = ["r", "g", "b", "m", "k", "Olive", "SaddleBrown", "CadetBlue", "DarkGreen", "Brown"]
    cnt = 0
    authors = {}
    new_labels = []
    for lbl in ylbls:
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
    ax.set_yticklabels(new_labels)

def result_filename():
    return title + sep + str(mfwords) + ' mfw. ' + delta_choice + sep + format_time()

def display_results(deltas):
    """
    creates a dendogram which is displayed and saved to a file
    there is a bug (I think) in the dendrogram method in scipy.cluster.hierarchy (probably)
    or in matplotlit. If you choose  orientation="left" or "bottom" the method get_text on the labels
    returns an empty string.
    """
    #clear the figure
    plt.clf()
    #create the datamodel which is needed as input for the dendrogram
    z = sch.linkage(deltas, method='ward', metric='euclidean')
    #create the dendrogram
    p = sch.dendrogram(z, orientation="left", labels=shorten_labels(deltas.index), link_color_func=lambda k: 'k')
    #get the axis
    ax = plt.gca()
    color_coding_author_names(ax)
    plt.title(title)
    plt.xlabel(str(mfwords) + " most frequent words. " + delta_choice)
    plt.tight_layout(2)
    plt.savefig(result_filename() + ".png")
    plt.show()


def save_results(mfw_corpus, deltas):
    """saves results to files
    :param mfw_corpus: the DataFrame shortened to the most frequent words
    :param deltas: a DataFrame containing the Delta distance between the texts
    """
    mfw_corpus.to_csv("corpus.csv", encoding="utf-8")
    deltas.to_csv(result_filename() + ".results.csv")


def main():
    corpus = process_files(encoding=encoding)
    mfw_corpus = preprocess_mfw_table(corpus)

    for distance_function in delta_algorithm.values():
        global delta_choice             # ugh.
        delta_choice = distance_function
        deltas = calculate_delta(mfw_corpus)
        display_results(deltas)
        save_results(mfw_corpus, deltas)
    print("Done.")


if __name__ == '__main__':
    #main() 
    cProfile.run('main()', "profile.txt")

