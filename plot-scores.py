#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from ggplot import *


scores = pd.DataFrame.from_csv("all-scores.csv")
ld     = scores[scores["Algorithm"] == "Linear_Delta" ]

linear = ggplot(aes(x="Words", y="Simple_Delta_Score", shape="Case_Sensitive", 
                     color="Corpus"), data=ld) \
          + geom_point() \
          + ylab("Scores") + ylim(0,2.7) \
          + ggtitle("Linear_Delta") \
          + theme_seaborn(context='paper')
ggsave(linear, "linear-delta-scores.pdf", width=29.7, height=20.5, units='cm')

deltas = ggplot(aes(x="Words", y="Simple_Delta_Score", shape="Case_Sensitive",
                    color="Corpus"), data=scores) \
         + geom_point() \
         + ylab("Scores") + ylim(0,2.7) \
         + facet_wrap("Algorithm") \
         + theme_seaborn(context='paper')
ggsave(deltas, "all-delta-scores.pdf", width=29.7, height=20.5, units="cm")

errors = ggplot(aes(x="Words", y="Clustering_Errors", shape="Case_Sensitive",
                    color="Corpus"), scores) \
         + geom_point() \
         + scale_y_reverse()  + ylab("Errors") + \
                          ylim(scores["Clustering_Errors"].min() - 2,
                               scores["Clustering_Errors"].max() + 1) \
         + facet_wrap("Algorithm") \
         + theme_seaborn(context='paper')
ggsave(errors, "all-delta-errors.pdf", width=29.7, height=20.5, units="cm")