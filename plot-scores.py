#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from ggplot import *
import os

try:
         os.mkdir("plots")
except FileExistsError:
         pass

scores = pd.DataFrame.from_csv("all-scores.csv")
algorithms = set(scores["Algorithm"].values)

for algo in algorithms:
         algo_scores = scores[scores["Algorithm"] == algo ]

         linear = ggplot(aes(x="Words", y="Simple_Delta_Score", shape="Case_Sensitive", 
                              color="Corpus"), data=algo_scores) \
                   + geom_point() \
                   + ylab("Scores") + ylim(0,2.7) \
                   + ggtitle(algo) \
                   + theme_seaborn(context='paper')
         ggsave(linear, "plots/delta-scores-{}.pdf".format(algo))

deltas = ggplot(aes(x="Words", y="Simple_Delta_Score", shape="Case_Sensitive",
                    color="Corpus"), data=scores) \
         + geom_point() \
         + ylab("Scores") + ylim(0,2.7) \
         + facet_wrap("Algorithm") \
         + theme_seaborn(context='paper')
ggsave(deltas, "plots/all-delta-scores.pdf", width=29.7, height=20.5, units="cm")

errors = ggplot(aes(x="Words", y="Clustering_Errors", shape="Case_Sensitive",
                    color="Corpus"), scores) \
         + geom_point() \
         + scale_y_reverse()  + ylab("Errors") + \
                          ylim(scores["Clustering_Errors"].min() - 2,
                               scores["Clustering_Errors"].max() + 1) \
         + facet_wrap("Algorithm") \
         + theme_seaborn(context='paper')
ggsave(errors, "plots/all-delta-errors.pdf", width=29.7, height=20.5, units="cm")
