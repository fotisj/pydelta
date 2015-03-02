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
import randomize_texts as rt

def filename(textcount, fname, mfw, lower_case):
    return "{}.{:04}.{}.{:03}.csv".format(
        fname.title(),
        mfw,
        "case_insensitive" if lower_case else "case_sensitive",
        textcount)

def parse_range(rangespec):
    """
    Creates a list of integers from a range specification in string form.

    Syntax:

        rangespec  ::= singlespec ("," singlespec )*
        singlespec ::= start [":" stop [":" step]]

    start, stop, step are arguments to :func:`range`. E.g.,
    `100,500:2001:500,3000` yields `[100, 500, 1000, 1500, 2000, 3000]`
    """
    if isinstance(rangespec, list):
        rangespec = ",".join(rangespec)
    return list(chain.from_iterable(
             range(*map(int, part.split(":"))) if ":" in part 
             else [int(part)] 
             for part in rangespec.split(",")))

def sweep(corpus_dir='corpus',
          refcorpus_dir=None,
          output='target',
          overwrite=False,
          cont=False,
          mfws = [100, 500, 1000, 1500, 2000, 2500, 3000, 5000],
          frequency_table=None,
          words = None,
          randomize_texts = None,
          randomize_method = 'authorsfirst'
          ):
    """
    """
    if words is not None:
        mfws = parse_range(words)

    print("MFW counts: ", *mfws)

    if randomize_texts:
        randomize_texts = parse_range(randomize_texts)
        randomize_func  = rt.randomizations[randomize_method]
        textlist = randomize_func(rt.list_subdir(corpus_dir))
    else:
        textlist = None
        randomize_texts = [ len(rt.list_subdir(corpus_dir)) ]

    print("Text counts:", *randomize_texts)

    # The score df will be used to store the simple delta scores for each
    # combination as a rough first guide. This will be dumped to a CSV
    # file after at the end.
    score_index = pd.MultiIndex.from_product([randomize_texts, 
            [name.title() for name in const.__dict__.keys()],
            mfws,
            [False, True]], names=["NTexts", "Algorithm", "Words", "Case Insensitive"])
    scores = pd.DataFrame(index=score_index, columns=["Score"])

    try:
        os.mkdir(output)
    except FileExistsError:
        if not (overwrite or cont):
            print("ERROR: Output folder {} already exists. Force overwriting using -f".format(output))
            return

    if textlist:
        textlist_name = os.path.join(output, 'texts.txt')
        if cont and os.path.exists(textlist_name):
            print("Reading already randomized text list from", textlist_name)
            with open(textlist_name, 'rt') as f:
                textlist = [l.strip() for l in f]
        else:
            print("Saving randomized text list to", textlist_name)
            with open(textlist_name, 'wt') as f:
                f.writelines(l + '\n' for l in textlist)

    evaluate = Eval()

    for textcount in randomize_texts:

        print("Running experiments with {} texts ...".format(textcount))

        if frequency_table is None or len(frequency_table) == 0:
            # Prepare the raw corpora
            filelist = textlist[0:textcount] if textlist else None
            corpora = [Corpus(subdir=corpus_dir, filelist=filelist),
                       Corpus(subdir=corpus_dir, lower_case=True, filelist=filelist)]
            cases = (False, True)
        else:
            ft = pd.read_table(frequency_table[0], sep=" ", index_col=0).fillna(0) / 100
            corpora = [None, Corpus(corpus=ft)]
            cases = (True,)

        if refcorpus_dir is not None:
            refcorpora = [Corpus(subdir=refcorpus_dir).get_mfw_table(0),
                          Corpus(subdir=refcorpus_dir, lower_case=True).get_mfw_table(0)]

        for mfw in mfws:
            for fname, fno in const.__dict__.items():
                for lc in cases:
                    outfn = os.path.join(output, filename(textcount, fname, mfw, lc))
                    if (cont and os.path.isfile(outfn)):
                        print("Skipping {}: it already exists".format(outfn))
                    else:
                        print("Preparing", outfn, "... ", end='')
                        c_mfw = corpora[lc].get_mfw_table(mfw)
                        if fno == const.ROTATED_DELTA:
                            if refcorpus_dir is None:
                                refc = corpora[lc]
                            else:
                                refc = refcorpora[lc]
                        else:
                            refc = None
                        delta = Delta(c_mfw, fno, refcorpus=refc)
                        score = evaluate.evaluate_deltas(delta, verbose=False)
                        print(score)
                        scores.loc[textcount, fname.title(), mfw, lc] = score
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
    parser.add_argument('-t', '--frequency-table', nargs=1, action="store",
            help="File with frequency tables from stylo, ignore corpus_dir")
    parser.add_argument('-n', '--randomize-texts', nargs=1, default=None, metavar="SPEC",
                        help="""
                        Sweep over a subset of texts of varying size. The argument is a spec 
                        as for the --words option. If given, a random list of the texts in
                        the corpus is created and we iterate over the first n texts, with n
                        according to this spec.
                        """)
    parser.add_argument('-m', '--randomize-method', choices=rt.randomizations, default='authorsfirst', help="Randomization method for variable number of texts")
    
    options = parser.parse_args()
    if options.output is None:
        options.output = options.corpus_dir + "_deltas"
        
    sweep(**options.__dict__)
