#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:35:11 2014

@author: vitt
"""

# wohl das häßlicste script ever

import argparse
import pandas as pd
import os

args = argparse.ArgumentParser(description="Convert a bunch of delta csvs to one file")
args.add_argument("deltas", help="directory containing the delta csv files")
args.add_argument("--verbose", "-v", help="be verbose", action='store_true')
options = args.parse_args()


def unstack(options):
    for filename in os.listdir(options.deltas):
        try:
            alg, word_s, case_s, _ = filename.split('.')
            words = int(word_s, 10)
            case_sensitive = case_s == 'case_sensitive'
            
            if options.verbose:
                print("Reading {} ...".format(filename))
            
            crosstab = pd.DataFrame.from_csv(os.path.join(options.deltas, filename))
            
            deltas = crosstab.unstack().to_frame() # -> MultiIndex + 1 unlabeled column
            deltas.columns.name = "Delta" # ugh, must be improvable
            deltas.index.names = [ "File1", "File2" ]
            deltas["Algorithm"] = alg
            deltas["Words"] = words
            deltas["Case Sensitive"] = case_sensitive
            
            # must be improvable as well
            deltas["Author1"] = deltas.index.to_series().map(lambda t: t[0].split('_')[0])
            deltas["Title1"] = deltas.index.to_series().map(lambda t: t[0].split('_')[1][:-4]) #strip .txt
            deltas["Author2"] = deltas.index.to_series().map(lambda t: t[1].split('_')[0])
            deltas["Title2"] = deltas.index.to_series().map(lambda t: t[1].split('_')[1][:-4]) 
            
            yield deltas
        except ValueError:
            print("WARNING: Skipping non-matching filename {}".format(filename))

if options.verbose:
    print("Processing {} ...".format(options.deltas))

bigtable = pd.concat(unstack(options))

if options.verbose:
    print("Saving results for {} ...".format(options.deltas))

bigtable.to_csv(options.deltas + '.csv')
