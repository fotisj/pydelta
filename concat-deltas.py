#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:35:11 2014

@author: vitt
"""

# wohl das häßlicste script ever

import matplotlib as mpl
mpl.use('Agg')

import argparse
import pandas as pd
import os
import delta
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn import metrics

STYLO_ALGS = {
       "AL": "Linear_Delta",
       "CB": "Canberra",
       "CD": "Classic_Delta",
       "ED": "Eders_Delta",
       "ES": "Eders_Simple_Delta",
       "EU": "Euclidean",
       "MH": "Manhattan"
}

args = argparse.ArgumentParser(description="Convert a bunch of delta csvs to one file")
args.add_argument("deltas", help="directories containing the delta csv files", nargs='+')
args.add_argument("-v", "--verbose", help="be verbose", action='store_true')
args.add_argument("-e", "--evaluate", help="evaluate each delta matrix", action='store_true')
args.add_argument("-a", "--all", nargs=1, default="", 
        help="Also concatenate all subcorpora and same them to the given file.")
args.add_argument("-p", "--pickle", action="store_true",
        help="The raw deltas will be pickled instead of stored as csv")
args.add_argument("-c", "--case-sensitive", action="store_true", default=False, 
        help="""When reading stylo written difference tables, assume they are
                for case-sensitive data. The default is case-insensitive.""")
args.add_argument("-d", "--dendrograms", nargs=1,
        help="Generate dendrograms and store them in the given directory.")
options = args.parse_args()

def progress(msg='.', end=''):
    """
    Prints the given progress message iff we want verbose output.
    """
    if options.verbose:
        print(msg, flush=True, end=end)

def corpus_name(dirname):
    """
    Converts the given dirname to a corpus name.
    """
    if dirname.startswith("deltas-"):
        return dirname[7:]
    else:
        return dirname

def cluster_errors_2(delta, z=None):
    """
    Calculates the number of cluster errors by:
    1. calculating the total number of different authors in the set
    2. calling sch.fcluster to generate at most that many flat clusters 
    3. for each of those clusters, the cluster errors are the number of authors in this cluster - 1
    4. sum of each cluster's errors = result
    """
    if z is None:
        z = sch.ward(delta)

    clusters = pd.DataFrame(index=delta.index)
    clusters["Author"] = [ s.split("_")[0] for s in clusters.index ]
    def nauthors(df):
        """Number of different authors in that df"""
        return len(set(df.Author))
    author_count = nauthors(clusters)
    clusters["Cluster"] = sch.fcluster(z, author_count, criterion='maxclust')
    return int((clusters.groupby("Cluster").agg(nauthors)-1).sum())

def adjusted_rand_index(delta, z):
    """
    Calculates the Adjusted Rand Index for the given delta and linkage matrix.
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score
    """

    def author(author_work):
        return author_work.split('_')[0]

    # Ground truth: Since ARI works with permutations, we can just assign each
    # author an arbitrary number. We just need to assign the same number to all
    # works of the same author.
    works = delta.index
    authors = { author(work) for work in works }
    author_idx = dict(zip(authors, range(0, len(authors))))
    truth = [ author_idx[author(work)] for work in works ]
    clusters = sch.fcluster(z, len(authors), criterion='maxclust')
    ari = metrics.adjusted_rand_score(truth, clusters)
    return ari

def _color_coding_author_names(ax, fig_orientation):
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


def read_directory(directory, evaluate=True):
    """
    Reads the delta crosstables in the given directory into one big dataframe.
    If evaluate is true, also performs two metrics on each delta table --
    clustering error counting and the simple delta score. 

    Returns a pair of dataframes, delta table first, scores second. If
    evaluate=False, the second return value is meaningless.
    """

    ev = delta.Eval()
    scores = pd.DataFrame(columns=["Algorithm", "Words", "Case_Sensitive", "Corpus",
        "Simple_Delta_Score", "Clustering_Errors", "Errors2", "Adjusted_Rand_Index"])
    scores.index.name = 'deltafile'
    corpus = corpus_name(directory)

    progress("\nProcessing directory {} (= corpus {})\n".format(directory, corpus))


    def unstack():    
        colors = ["r", "g", "b", "m", "k", "Olive", "SaddleBrown", "CadetBlue", "DarkGreen", "Brown"]
        for filename in sorted(os.listdir(directory)):
            try:
                try:
                    alg, word_s, case_s, _ = filename.split('.')
                    case_sensitive = case_s == 'case_sensitive'
                except ValueError:
                    alg, word_s, _ = filename.split('.')
                    if alg in STYLO_ALGS:
                        alg = STYLO_ALGS[alg]
                        case_sensitive=options.case_sensitive
                    else:
                        print("{} does not match, and {} is not one of the stylo distances".format(filename, alg))
                        raise

                words = int(word_s, 10)
                
                progress("Reading {} ".format(filename))
                crosstab = pd.DataFrame.from_csv(os.path.join(directory, filename))
                progress()
                
                if evaluate:
                    simple_score = ev.evaluate_deltas(crosstab, verbose=False)
                    progress()
                    linkage = sch.ward(crosstab)
                    plt.clf()
                    fig = plt.gcf()
                    fig.set_size_inches(8.07,11.69)
                    fig.set_dpi(600)
                    dendrogram = sch.dendrogram(linkage, labels=[fn[:-4] for fn in crosstab.index],
                            leaf_font_size=8, link_color_func=lambda k: 'k', orientation="left")
                    progress()
                    total, errors = ev.evaluate_results(dendrogram)
                    errors2 = cluster_errors_2(crosstab, linkage)
                    ari = adjusted_rand_index(crosstab, linkage)
                    if options.dendrograms:
                        plt.title("{} {} CS: {} mfw {}".format(corpus, alg, case_sensitive, words))
                        plt.xlabel("Errors {}, Score {}".format(errors, simple_score))
                        ax = plt.gca()
                        _color_coding_author_names(ax, "left")
                        plt.tight_layout()
                        plt.savefig(os.path.join(options.dendrograms[0], 
                            corpus + "." + filename[:-3] + "pdf"), 
                            dpi=600,
                            orientation="portrait", papertype="a4", 
                            format="pdf") 
                    scores.loc[filename] = (alg, words, case_sensitive, corpus, simple_score, errors, errors2, ari)
                    progress()
                else:
                    progress("...")
                
                deltas = crosstab.unstack().to_frame() # -> MultiIndex + 1 unlabeled column
                deltas.columns.name = "Delta" # ugh, must be improvable
                deltas.index.names = [ "File1", "File2" ]
                deltas["Algorithm"] = alg
                deltas["Words"] = words
                deltas["Case_Sensitive"] = case_sensitive
                deltas["Corpus"] = corpus
                
                # must be improvable as well
                deltas["Author1"] = deltas.index.to_series().map(lambda t: t[0].split('_')[0])
                deltas["Title1"] = deltas.index.to_series().map(lambda t: t[0].split('_')[1][:-4]) #strip .txt
                deltas["Author2"] = deltas.index.to_series().map(lambda t: t[1].split('_')[0])
                deltas["Title2"] = deltas.index.to_series().map(lambda t: t[1].split('_')[1][:-4]) 
                progress("\n")
                yield deltas
            except ValueError(e):
                print("WARNING: Skipping non-matching filename {}".format(filename),e)

    corpus_deltas = pd.concat(unstack())
    return (corpus_deltas, scores)


all_deltas = None
all_scores = None

for directory in options.deltas:
    (deltas, scores) = read_directory(directory)
    progress("Saving deltas for {} ...".format(directory))
    if options.pickle:
        deltas.to_pickle(directory + ".pickle")
    else:
        deltas.to_csv(directory + ".csv")
    progress("\n")

    if options.evaluate:
        progress("Saving scores for {} ...\n".format(directory))
        scores.to_csv(directory + "-scores.csv")

    if options.all:
        if all_deltas is None:
            all_deltas = deltas
        else:
            all_deltas = pd.concat([all_deltas, deltas])

    if options.evaluate:
        if all_scores is None:
            all_scores = scores
        else:
            all_scores = pd.concat([all_scores, scores])

if options.evaluate:
    progress("Saving all scores to all-scores.csv")
    all_scores.to_csv("all-scores.csv")
    progress("\n")

if options.all:
    progress("Saving all deltas to {}".format(options.all))
    if options.pickle:
        all_deltas.to_pickle(options.all)
    else:
        all_deltas.to_csv(options.all)
    progress("\n")
