#!/usr/bin/python3

import pandas as pd
import delta
import evaluate
import argparse
import os


def get_argparser():
    parser = argparse.ArgumentParser(
        description="Creates delta tables for text length evaluation")
    parser.add_argument(
        'corpus_dir',
        help="A directory containing the corpus to process")
    parser.add_argument('-o', '--output', default='target',
                        help="Target directory for the delta CSVs. (%(default)s")
    parser.add_argument(
        '-w',
        '--words',
        action="store",
        default=2500,
        type=int,
        help="Number of most frequent words (%(default)s)")
    parser.add_argument(
        '-a',
        '--algorithm',
        action='store',
        default='COSINE_DELTA',
        help="Algorithm (%(default)s). One of " +
        ", ".join(
            delta.const.__dict__.keys()))
    parser.add_argument(
        '-M',
        '--max-chars',
        action="store",
        default=None,
        metavar="rangespec",
        help="""Limit the number of characters that is parsed.

                        Argument is a sequence of range spec, separated by comma. A range spec has the form min:max:step.
                        E.g., 15:26:5,50:251:50 means [15, 20, 25, 50, 100,
                        150, 200, 250].
                        """)
    parser.add_argument(
        '-L',
        '--max-chars-files',
        action="store",
        default=None,
        metavar="PATTERN",
        help="""If specified, the --max-chars argument is only used for files
                        that match the PATTERN (shell glob pattern, e.g., '*Nachtwachen*'""")
    return parser


def sweep(options):
    if options.max_chars is not None:
        max_chars = evaluate.parse_range(options.max_chars)
    else:
        max_chars = [None]

    alg = delta.const.__dict__[options.algorithm.upper()]
    os.makedirs(options.output, exist_ok=True)


    for max_c in max_chars:
        corpus = delta.Corpus(subdir=options.corpus_dir, max_chars=max_c,
                              max_chars_only=options.max_chars_files)
        mfw = corpus.get_mfw_table(options.words)
        deltas = delta.Delta(mfw, alg)
        outf = os.path.join(
            options.output,
            evaluate.filename(
                max_c,
                options.algorithm.upper(),
                options.words,
                False))
        deltas.to_csv(outf)


def main():
    options = get_argparser().parse_args()
    sweep(options)


if __name__ == '__main__':
    main()
