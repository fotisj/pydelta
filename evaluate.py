#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs a delta analysis on a number of parameter combinations.

@author: Thorsten Vitt <thorsten.vitt@uni-wuerzburg.de>
"""

from delta import *
import os
import pandas as pd

def filename(fname, mfw, lower_case):
    return "target/{}.{:04}.{}.csv".format(
        fname.title(),
        mfw,
        "case_insensitive" if lower_case else "case_sensitive")


def main():
    
    mfws = [100, 500, 1000, 1500, 2000, 2500, 3000, 5000]
    score_index = pd.MultiIndex.from_product([[name.title() for name in const.__dict__.keys()], 
            mfws,
            [False, True]], names=["Algorithm", "Words", "Case Insensitive"])
    scores = pd.DataFrame(index=score_index, columns=["Score"])
    
    try:
        os.mkdir("target")
    except FileExistsError:
        pass
    
    evaluate = Eval()

    # Prepare the raw corpora
    corpora = [Corpus(subdir='corpus2'), 
               Corpus(subdir='corpus2', lower_case=True)]
    
    for fname, fno in const.__dict__.items():
        for mfw in mfws:
            for lc in False, True:
                print("Preparing", filename(fname, mfw, lc), "... ", end='')
                c_mfw = corpora[lc].get_mfw_table(mfw)
                if fno == const.ROTATED_DELTA or fno == const.MAHALANOBIS:
                    refc = corpora[lc]
                else:
                    refc = None
                delta = Delta(c_mfw, fno, refcorpus=refc)
                score = evaluate.evaluate_deltas(delta, verbose=False)
                print(score)
                scores.loc[fname.title(), mfw, lc] = score
                delta.to_csv(filename(fname, mfw, lc))
        # dump the scores after every alg at least
        scores.to_csv("target/delta-scores.csv")

if __name__ == '__main__':
    main()
