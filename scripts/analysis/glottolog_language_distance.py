#!/usr/bin/env python

# Loads TSV file that contains the following columns:
# * Language
# * Classification (slash-separated list of language families)
#
# Computes the distance between each pair of languages, defined as
# the number of steps required to get from one language to the other
# in the classification tree.

import itertools as it
import json
from collections import defaultdict
from functools import partial

import click
import pandas as pd
from rich.console import Console
from rich.progress import track

ISO_COLUMN_NAME = "ISO639P3code"
LANG_COLUMN_NAME = "Name"
CLASS_COLUMN_NAME = "Classification"

COLUMNS = [ISO_COLUMN_NAME, LANG_COLUMN_NAME, CLASS_COLUMN_NAME]
STDERR_CONSOLE = Console(stderr=True)


def language_pair_distance(language_pair, iso_to_class):
    """Computes the distance between two languages.

    Args:
        language_pair (tuple): A tuple of ISO codes for the two languages.
        iso_to_class (dict): A mapping from ISO codes to the classification of
            the language.

    Returns:
        int: The distance between the two languages.
    """
    iso1, iso2 = language_pair

    class1 = iso_to_class[iso1]
    class2 = iso_to_class[iso2]

    # Find the first index where the two classifications differ

    for i, (c1, c2) in enumerate(zip(class1, class2)):
        if c1 != c2:
            break

    # The distance is the number of steps required to get from one language
    # to the other in the classification tree

    return len(class1) + len(class2) - 2 * i


@click.command()
@click.argument("tsv_file", type=click.Path(exists=True))
def main(tsv_file):
    tsv = pd.read_csv(tsv_file, sep="\t")

    # Remove rows with missing values
    tsv = tsv.dropna()

    # Create ISO-code => classification mapping
    iso_to_class = {
        iso: cls

        for iso, cls in zip(tsv[ISO_COLUMN_NAME], tsv[CLASS_COLUMN_NAME].str.split("/"))
    }

    # Create all language pairs
    language_pairs = list(it.combinations(iso_to_class.keys(), 2))
    compute_distance = partial(language_pair_distance, iso_to_class=iso_to_class)

    tsv_rows = ["lang1\tlang2\tdistance"]
    with click.get_text_stream("stdout") as fout:
        for iso1, iso2 in track(
            language_pairs,
            description="Computing distances (a, b)",
            total=len(language_pairs),
            # output to stderr
            console=STDERR_CONSOLE,
        ):
            dist = compute_distance((iso1, iso2))
            row = f"{iso1}\t{iso2}\t{dist}"
            tsv_rows.append(row)

        click.echo("\n".join(tsv_rows), file=fout)


if __name__ == "__main__":
    main()
