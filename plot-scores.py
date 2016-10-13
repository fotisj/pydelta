#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from ggplot import *
import argparse
import os

parser = argparse.ArgumentParser("Plot from a bunch of all-scores.csv files")
parser.add_argument("scores", default="all-scores.csv",
                  help="File containing the scores to plot")
parser.add_argument("-o", "--output-dir", default="plots",
                  help="output directory for the plots")
options = parser.parse_args()


print("Plotting scores from", options.scores, "to", options.output_dir)

try:
         os.mkdir(options.output_dir)
except FileExistsError:
         print("WARNING: Overwriting existing plots")
         pass

scores = pd.DataFrame.from_csv(options.scores)
algorithms = set(scores["Algorithm"].values)

for algo in algorithms:
         algo_scores = scores[scores["Algorithm"] == algo ]

         linear = ggplot(aes(x="Words", y="Simple_Delta_Score", shape="Case_Sensitive",
                              color="Corpus"), data=algo_scores) \
                   + geom_point(alpha=.7) \
                   + ylab("Scores") + ylim(0,4.0) \
                   + ggtitle(algo) \
                   + theme_seaborn(context='paper')
         linear.save(os.path.join(options.output_dir, "delta-scores-{}.pdf".format(algo)))

         err = ggplot(aes(x="Words", y="Clustering_Errors", shape="Case_Sensitive",
                    color="Corpus"), algo_scores) \
                  + geom_point(alpha=.7) \
                  + scale_y_reverse()  + ylab("Errors") + \
                                   ylim(scores["Clustering_Errors"].min() - 2,
                                        scores["Clustering_Errors"].max() + 1) \
                  + theme_seaborn(context='paper')
         err.save(os.path.join(options.output_dir, "delta-errors-{}.pdf".format(algo)))

         ari = ggplot(aes(x="Words", y="Adjusted_Rand_Index", shape="Case_Sensitive",
                    color="Corpus"), algo_scores) \
                  + geom_point(alpha=.7) \
                  + scale_y_reverse()  + ylab("Adjusted Rand Index") + \
                                   ylim(-1, 1) \
                  + theme_seaborn(context='paper')
         err.save(os.path.join(options.output_dir, "delta-ari-{}.pdf".format(algo)))


deltas = ggplot(aes(x="Words", y="Simple_Delta_Score", shape="Case_Sensitive",
                    color="Corpus"), data=scores) \
         + geom_point(alpha=.7,size=5) \
         + ylab("Scores") + ylim(0,4.0) \
         + facet_wrap("Algorithm") \
         + theme_seaborn(context='paper')
deltas.save(os.path.join(options.output_dir, "all-delta-scores.pdf"), width=29.7, height=20.5, units="cm")

errors = ggplot(aes(x="Words", y="Clustering_Errors", shape="Case_Sensitive",
                    color="Corpus"), scores) \
         + geom_point(alpha=.7,size=5) \
         + scale_y_reverse()  + ylab("Errors") + \
                          ylim(scores["Clustering_Errors"].min() - 2,
                               scores["Clustering_Errors"].max() + 1) \
         + facet_wrap("Algorithm") \
         + theme_seaborn(context='paper')
errors.save(os.path.join(options.output_dir, "all-delta-errors.pdf"), width=29.7, height=20.5, units="cm")

ari = ggplot(aes(x="Words", y="Adjusted_Rand_Index", shape="Case_Sensitive",
                    color="Corpus"), scores) \
         + geom_point(alpha=.7,size=5) \
         + ylab("Adjusted Rand Index") + \
                          ylim(-1, 1) \
         + facet_wrap("Algorithm") \
         + theme_seaborn(context='paper')
ari.save(os.path.join(options.output_dir, "all-delta-ari.pdf"), width=29.7, height=20.5, units="cm")
