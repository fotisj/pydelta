"""
Feature selection utilities.
"""



from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold

import numpy as np
import sklearn.svm


def get_rfe_features(corpus, estimator=None,
                     steps=[(10000, 1000), (1000, 200), (500, 25)], cv=True):
    """
    Args:
        corpus: containing document_describer,
        estimator: supervised learning estimator,
        steps: list of tuples (features_to_select, step)
        cv: additional cross-validated selection.
    Returns:
        rfe_terms: set of selected terms.
    """
    if estimator is None:
        estimator = sklearn.svm.SVC(kernel="linear")
    matrix = corpus
    groups = np.array([corpus.document_describer.group_name(x)
                       for x in corpus.index])
    terms = np.array(corpus.columns)
    for step in steps:
        rfe = RFE(estimator=estimator, n_features_to_select=step[0],
                  step=step[1])
        matrix = rfe.fit_transform(matrix, groups)
        terms = terms[rfe.support_]

    # cross-validation
    if cv:
        #rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(groups, n_folds=3), scoring='accuracy')
        rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(3), scoring='accuracy')
        rfecv.fit(matrix, groups)
    rfe_terms = terms[rfecv.support_]
    return set(rfe_terms)
