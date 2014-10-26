#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs a delta analysis on a number of parameter combinations.

@author: Thorsten Vitt <thorsten.vitt@uni-wuerzburg.de>
"""

from delta import *
import os
import pandas as pd
import argparse
from itertools import chain

def filename(fname, mfw, lower_case):
    return "{}.{:04}.{}.csv".format(
        fname.title(),
        mfw,
        "case_insensitive" if lower_case else "case_sensitive")


def sweep(corpus_dir='corpus',
          refcorpus_dir=None,
          output='target',
          overwrite=False,
          cont=False,
          mfws = [100, 500, 1000, 1500, 2000, 2500, 3000, 5000],
          words = None):
    """
    """
    if words is not None:
        if isinstance(words, list):
            words = ",".join(words)            
        mfws = list(chain.from_iterable( 
                 range(*map(int, part.split(":"))) if ":" in part 
                 else [int(part)] 
                 for part in words.split(",")))

    print("MFW counts: ", *mfws)
        
    # The score df will be used to store the simple delta scores for each
    # combination as a rough first guide. This will be dumped to a CSV
    # file after at the end.
    score_index = pd.MultiIndex.from_product([[name.title() for name in const.__dict__.keys()],
            mfws,
            [False, True]], names=["Algorithm", "Words", "Case Insensitive"])
    scores = pd.DataFrame(index=score_index, columns=["Score"])

    try:
        os.mkdir(output)
    except FileExistsError:
        if not (overwrite or cont):
            print("ERROR: Output folder {} already exists. Force overwriting using -f".format(output))
            return

    evaluate = Eval()

    # Prepare the raw corpora
    corpora = [Corpus(subdir=corpus_dir),
               Corpus(subdir=corpus_dir, lower_case=True)]

    for mfw in mfws:
        for fname, fno in const.__dict__.items():
            for lc in False, True:
                outfn = os.path.join(output, filename(fname, mfw, lc))
                if (cont and os.path.isfile(outfn)):
                    print("Skipping {}: it already exists".format(outfn))
                else:
                    print("Preparing", outfn, "... ", end='')
                    c_mfw = corpora[lc].get_mfw_table(mfw)
                    if fno == const.ROTATED_DELTA:
                        refc = corpora[lc]
                    else:
                        refc = None
                    delta = Delta(c_mfw, fno, refcorpus=refc)
                    score = evaluate.evaluate_deltas(delta, verbose=False)
                    print(score)
                    scores.loc[fname.title(), mfw, lc] = score
                    delta.to_csv(outfn)
        # dump the scores after every alg at least
        scores.to_csv(output + "_scores.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description="Create a bunch of delta tables for a corpus.",
                epilog="The script will write a file OUTPUT.csv containing the simple scores.")
    parser.add_argument('corpus_dir', default='corpus',
                        help="A directory containing the corpus to process")
    parser.add_argument('-r', '--refcorpus', dest='refcorpus_dir',
                        help="A directory containing the reference corpus.")
    parser.add_argument('-o', '--output',
                        help="Target directory for the delta CSVs.")
    parser.add_argument('-f', '--overwrite', action='store_true', default=False,
                        help='Overwrite target directory if neccessary')
    parser.add_argument('-c', '--continue', action='store_true', default=False, dest='cont',
                        help='Skip experiments when the respective output file exists')
    parser.add_argument('-w', '--words', nargs=1, default='100,500:3001:500,5000',
                        help="""
                        Numbers of most frequent words to try. The argument is a comma-separated
                        list of items. Each item is either an integer number (the number of words
                        to use) or a range expression in the form min:max:step, meaning take all
                        numbers from min increasing by step as long as they are lower than max.
                        """)
    
    options = parser.parse_args()
    if options.output is None:
        options.output = options.corpus_dir + "_deltas"
        
    sweep(**options.__dict__)
