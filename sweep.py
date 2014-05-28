#!/usr/bin/env python3

from delta import *
import pandas as pd
import os


def sweep():
    DROPPED = "dropped E* columns"

    os.makedirs("deltas", exist_ok=True)

    wordcounts = list(range(1,25)) + list(range(25,100,5)) + list(range(100,500,10)) + list(range(500,3050,50))

    cols = list(vars(const).keys()).append(DROPPED)

    eval_results = pd.DataFrame(index=wordcounts,columns=vars(const).keys())
    eval_results.index.name = 'MFWords'
    ev = Eval()

    print(dropped)

    corpus = Corpus('corpus')
    refcorpus = Corpus('refcorpus')

    for wordcount in wordcounts:
        print("\n\n# %s words\n" % wordcount)
        mfw_corpus = corpus.get_mfw_table(wordcount)
        
        for method in vars(const).keys():
            print(method, end=': ')
            deltas = Delta(mfw_corpus, const.__dict__[method], refcorpus) #XXX?
            quality = ev.evaluate_deltas(deltas, verbose=False)
            print(quality)
            eval_results.at[wordcount, method] = quality
            if method == "ROTATED_DELTA":
                eval_results.at[wordcount, DROPPED] = dropped
            deltas.to_csv("deltas/{words:04d}.{method}.csv".format(method=method, words=wordcount))

    eval_results.to_csv("qualities_corpus3_large_refcorpus_dropped.csv")

if __name__ == "__main__":
    sweep()
