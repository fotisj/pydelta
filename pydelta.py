#!/usr/bin/env python3 
"""
calculates Burrow's Delta and Argamon's proposed variations
tbd:
- add evaluation for the results. 
- write a gui to set all the configuration information

"""
import sys
import glob
import re
import collections
import os
import pandas as pd
import csv
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import matplotlib.pylab as plt
import itertools
from datetime import datetime
import profig
import logging


def get_configuration():
    """
    writes/reads a default configuration to pydelta.ini. If you want to change these parameters, use the ini file
    """
    config = profig.Config("pydelta.ini")

    #where to find the corpus
    config.init("files.subdir", "corpus", comment="the subdirectory containing the text files used as input")

    #file_encoding
    config.init("files.encoding", "utf-8", comment="the file encoding for the input files")

    #use existing wordlist #  'yes' not implemented yet
    config.init("files.use_wordlist", False, comment="not implemented yet")

    #sample size, only useful if set_limit is True
    config.init("data.limit", 200000, comment="amount of file to be processed in chars. only useful in "
                                              "combination with set_limit")

    #switch to extract only a sample. NOT tested yet
    config.init("data.set_limit", False, comment="only use a specified amount of the file. NOT TESTED")

    #all words in lower case?
    config.init("data.lower_case", False, comment="convert all words to lower case")

    #name of the file with the complete wordlist is saved
    config.init("save.fileout", "corpus_words.csv", comment="sets the file name " +
                                                            "to save the corpus words into a file")

    #save deltas to file results.csv
    config.init("save.save_results", True, comment="the results (i.e. the delta measures) are written " +
                "into the file results.csv")

    #separates information in filename
    config.init("save.sep", "-", comment="sperates time and statistic information in the image filename")

    #number of words to use from the wordlist
    #set to 0 to use all
    config.init("statistics.mfwords", 2000, comment="number of most frequent words to use " +
                                                    "in the calculation of delta")

    #name of the chosen algorithm
    #constraint of the profig modul: it doesn't handle dict, so
    #we have to use a list here
    config.init("statistics.delta_algorithm", ["Classic Delta",
                                               "Argamon's linear Delta",
                                               "Argamon's quadratic Delta"], comment="available delta algorithms")

    #possible values are 0,1,2
    config.init("statistics.delta_choice", 0, comment="choice of the delta algorithm. 0 = classic, 1 = linear usw.")

    config.init("statistics.linkage_method", "ward", comment="method how the distance between the newly formed " +
                                                             "cluster and each candidate is calculated. Valid " +
                                                             "valuues: 'ward', 'single', 'average', 'complete',   " +
                                                             "'weighted'. See documentation on " +
                                                             "scipy.cluster.hierarchy.linkage for details.")

    config.init("statistics.evaluate", True, comment="evaluation of the results. Only useful if there are always" +
                                                     "more than 2 texts by an author and the attribution of all " +
                                                     "texts is known.")

    #title in figure and filename
    config.init("figure.fig_title", '3 novelists from German realism', comment="title is used in the " +
                                                                               "figure and the name of the "+
                                                                               "file containing the figure")

    #use filenames as labels in plot
    config.init("figure.filenames_labels", True, comment="use the filename of the texts as labels")

    #orientation of the figure
    #allowed values are 'left' (author and title at the left side) and 'top' (author/title at the bottom)
    config.init("figure.fig_orientation", "left", comment="orientation of the figure. allowed values: 'left', 'top'")

    #font-size used in the figure
    config.init("figure.font_size", 11, comment="font-size for labels in the figure")

    #writes configuration files if it doesn't exist, otherwise reads in
    #values from there
    config.sync()

    return config


def process_files(config):
    """
    preprocessing all files ending with *.txt in corpus subdir
    all files are tokenized
    a table of all word and their freq in all texts is created
    format
            filename1 filename2 filename3
    word      nr        nr
    word      nr        nr
    :param: config: access to configuration settings
    """
    if not os.path.exists(config['files.subdir']):
        print("The directory " + config['files.subdir'] + " doesn't exist. \nPlease add a directory " +
                                                          "with text files.\n")
        sys.exit(1)
    filelist = glob.glob(config['files.subdir'] + os.sep + "*.txt")
    list_of_wordlists = []
    for file in filelist:
        list_of_wordlists.append(tokenize_file(file, config["files.encoding"], config))
    corpus_words = pd.DataFrame(list_of_wordlists).fillna(0)
    return corpus_words.T


def tokenize_file(filename, encoding, config):
    """
    tokenizes file and returns an unordered pandas.DataFrame
    containing the words and frequencies
    standard encoding = utf-8
    """
    all_words = collections.defaultdict(int)

    #read file, tokenize it, count words
    read_text_length = 0
    #reading the config information only once because of speed
    set_limit = config["data.set_limit"]
    limit = config["data.limit"]
    lower_case = config["data.lower_case"]
    with open(filename, "r", encoding=encoding) as filein:
        #print("processing " + filename)
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


def preprocess_mfw_table(corpus, config):
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
    if config["statistics.mfwords"] > 0:
        return new_corpus[:config["statistics.mfwords"]]
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


def calculate_delta(corpus, config):
    """
    chooses the algorithm for the calculation of delta
    after rewriting classic_delta this can be moved there
    """
    if config["statistics.delta_choice"] == 0:
        return classic_delta(corpus, config["statistics.mfwords"])
    elif config["statistics.delta_choice"] == 1:
        return linear_delta(corpus)
    elif config["statistics.delta_choice"] == 2:
        return quadratic_delta(corpus)
    else:
        #tbd: use raise Exception for the following
        print("ERROR: You have to choose an algorithm for Delta.")
        sys.exit(1)


def classic_delta(corpus, mfwords):
    """
    calculates Delta in the simplified form proposed by Argamon
    """
    #print("using classic delta")
    stds = corpus_stds(corpus)
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


def color_coding_author_names(ax, config):
    """color codes author names
    :param ax: a matplotlib axis as created by the dendogram method
    """
    lbls = []
    #get the labels from axis
    if config["figure.fig_orientation"] == "left":
        lbls = ax.get_ymajorticklabels()
    elif config["figure.fig_orientation"] == "top":
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
    if config["figure.fig_orientation"] == "left":
        ax.set_yticklabels(new_labels)
    elif config["figure.fig_orientation"] == "top":
        ax.set_xticklabels(new_labels)


def display_results(deltas, config):
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
    if config["statistics.linkage_method"] == "ward":
        z = sch.linkage(deltas, method='ward', metric='euclidean')
    else:
        #creating a flat representation of the dist matrix
        deltas_flat = ssd.squareform(deltas)
        z = sch.linkage(deltas_flat, method=config["statistics.linkage_method"], metric='euclidean')
    #print("linkage method: ", config["statistics.linkage_method"])
    #create the dendrogram
    if config["figure.fig_orientation"] == "top":
        rotation = 90
    elif config["figure.fig_orientation"] == "left":
        rotation = 0
    dendro_data = sch.dendrogram(z, orientation=config["figure.fig_orientation"], labels=shorten_labels(deltas.index),
                   leaf_rotation=rotation, link_color_func=lambda k: 'k', leaf_font_size=config["figure.font_size"])
    #get the axis
    ax = plt.gca()
    color_coding_author_names(ax, config)
    plt.title(config["figure.fig_title"])
    plt.xlabel(str(config["statistics.mfwords"]) + " most frequent words. " + config["statistics.delta_algorithm"][
        config["statistics.delta_choice"]])
    plt.tight_layout(2)
    plt.savefig(result_filename(config) + ".png")
    plt.show()
    return dendro_data


def format_time(config):
    """
    helper method. date and time are used to create a string for inclusion into the filename
    rtype: str
    """
    dt = datetime.now()
    return u'{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(str(dt.year), config["save.sep"], str(dt.month),
                                                 config["save.sep"], str(dt.day), config["save.sep"],
                                                 str(dt.hour), config["save.sep"], str(dt.minute))


def result_filename(config):
    """
    helper method to format the filename for the image which is saved to file
    the name contains the title and some info on the chosen statistics
    """
    return config["figure.fig_title"] + config["save.sep"] + str(config["statistics.mfwords"]) \
             + " mfw. " + config["statistics.delta_algorithm"][config["statistics.delta_choice"]] \
             + config["save.sep"] + format_time(config)


def save_results(mfw_corpus, deltas, config):
    """saves results to files
    :param mfw_corpus: the DataFrame shortened to the most frequent words
    :param deltas: a DataFrame containing the Delta distance between the texts
    """
    mfw_corpus.to_csv("corpus.csv", encoding="utf-8")
    deltas.to_csv(result_filename(config) + ".results.csv")


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
    checks whether the distance of a given text to texts of other authors is smaller than to
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
        if d == None:
            d = 0
        else:
            if l[i] - l[i-1] > 1:
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

    print ("attributions - errors: ", len(ivl), " - ", errors)

def main():
    config = get_configuration()
    corpus = process_files(config)
    mfw_corpus = preprocess_mfw_table(corpus, config)
    deltas = calculate_delta(mfw_corpus, config)
    fig = display_results(deltas, config)
    save_results(mfw_corpus, deltas, config)
    if config["statistics.evaluate"]:
        evaluate_results(fig)

if __name__ == '__main__':
    main()


