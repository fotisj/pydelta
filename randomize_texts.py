#!/usr/bin/env python3

import os
import sys
import argparse
import random
import glob
import collections

def simple_randomization(files):
    return random.sample(files, len(files))

def author_of(filename):
    return filename.split('_')[0]

def authors_first(files):
    # first, map by author
    byauthor = collections.defaultdict(list)
    for f in files:
        byauthor[author_of(f)].append(f)
    available = list(files) # files not used yet
    result = []             # randomized file list

    authors = random.sample(byauthor.keys(), len(byauthor))
    # now, for every author randomly pick two texts.
    for author in authors:
        texts = byauthor[author]
        chosen = random.sample(texts, 2)
        for c in chosen:
            result.append(c)
            available.remove(c)

    # now, we have all authors, and at least two texts per author
    # we can just randomly choose from the remainder
    result += random.sample(available, len(available))
    return result

randomizations = { 'sample': simple_randomization, 'authorsfirst': authors_first }

def list_subdir(subdir, pattern='*.txt'):
    """
    Lists all files matching `pattern` in the given `subdir`, returns a list of the basenames.
    """
    return [ os.path.basename(path)
             for path in glob.iglob(os.path.join(subdir, pattern)) ]

def main():
    parser = argparse.ArgumentParser(
                description="Create a bunch of delta tables for a corpus.",
                epilog="The script will write a file OUTPUT.csv containing the simple scores.")
    parser.add_argument('corpus_dir', default='corpus',
                        help="A directory containing the corpus to process")
    parser.add_argument('-o', '--output', 
                        help="Where to write the randomized file list")
    parser.add_argument('-m', '--method', choices=randomizations, default='authorsfirst',
                        help="Randomization method.")

    options = parser.parse_args()
    files = randomizations[options.method](list_subdir(options.corpus_dir))
    if options.output:
        with open(options.output, "w") as out:
            out.writelines(line + '\n' for line in files)
    else:
        sys.stdout.writelines(line + '\n' for line in files)

if __name__ == '__main__':
    main()
