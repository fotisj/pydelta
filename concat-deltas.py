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

args = argparse.ArgumentParser(description="Convert a bunch of delta csvs to one file")
args.add_argument("deltas", help="directories containing the delta csv files", nargs='+')
args.add_argument("-v", "--verbose", help="be verbose", action='store_true')
args.add_argument("-e", "--evaluate", help="evaluate each delta matrix", action='store_true')
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
        "Simple_Delta_Score", "Clustering_Errors"])
    scores.index.name = 'deltafile'
    corpus = corpus_name(directory)

    progress("\nProcessing directory {} (= corpus {})\n".format(directory, corpus))

    def unstack():    
        for filename in sorted(os.listdir(directory)):
            try:
                alg, word_s, case_s, _ = filename.split('.')
                words = int(word_s, 10)
                case_sensitive = case_s == 'case_sensitive'
                
                progress("Reading {} ".format(filename))
                crosstab = pd.DataFrame.from_csv(os.path.join(directory, filename))
                progress()
                
                if evaluate:
                    simple_score = ev.evaluate_deltas(crosstab, verbose=False)
                    progress()
                    linkage = sch.ward(crosstab)
                    plt.clf()
                    dendrogram = sch.dendrogram(linkage, labels=crosstab.index)
                    progress()
                    total, errors = ev.evaluate_results(dendrogram)
                    scores.loc[filename] = (alg, words, case_sensitive, corpus, simple_score, errors)                
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
            except ValueError:
                print("WARNING: Skipping non-matching filename {}".format(filename))

    corpus_deltas = pd.concat(unstack())
    return (corpus_deltas, scores)


all_deltas = None
all_scores = None

for directory in options.deltas:
    (deltas, scores) = read_directory(directory)
    progress("Saving deltas for {} ...".format(directory))
    deltas.to_csv(directory + ".csv")
    progress("\n")

    if options.evaluate:
        progress("Saving scores for {} ...\n".format(directory))
        scores.to_csv(directory + "-scores.csv")

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
    progress("Saving all scores to all-scores.csv\n")
    all_scores.to_csv("all-scores.csv")

progress("Saving all deltas to all-deltas.csv\n")
all_deltas.to_csv("all-deltas.csv")
