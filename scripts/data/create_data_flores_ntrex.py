import functools
from collections import defaultdict
from pathlib import Path

import click
import datasets as ds


@click.command()
@click.option(
    "--data-folder",
    type=click.Path(exists=True),
    required=True,
    help="Path to the folder containing the parallel data files.",
)
@click.option(
    "--output-folder", type=click.Path(),
    required=True,
    help="Path to the folder where the train and test datasets will be saved.",
)
@click.option("--corpus-name", type=click.Choice(["flores200", "ntrex"]), required=True)
@click.option(
    "--test-size",
    type=float,
    default=0.2,
    help="Fraction of the data to be used for testing.",
)
@click.option("--get-rid-of-orth", is_flag=True)
def create_dataset(
    data_folder,
    output_folder,
    corpus_name,
    test_size,
    get_rid_of_orth=False,
):
    # Get a list of all the parallel data files in the data folder
    file_names = list(
        Path(data_folder).glob(
            "*.dev" if corpus_name.startswith("flores") else "newstest*2019*.txt"
        )
    )

    # Load the data from each file into a list of dictionaries
    data = ds.DatasetDict()

    def add_lang(d, lang="fin"):
        d["language"] = lang

        return d

    FLORES_MODE = corpus_name.startswith("flores")

    for file_name in file_names:
        with open(file_name, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f]

            if FLORES_MODE:
                lang = file_name.name.replace(".dev", "")
            else:
                lang = file_name.name.split(".")[1]

            sep = "_" if FLORES_MODE else "-"
            lang = lang.split(sep)[0] if get_rid_of_orth else lang

            data[lang] = ds.Dataset.from_dict({"text": sentences}).map(
                functools.partial(add_lang, lang=lang)
            )

    # Save the train and test datasets to disk
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    data.save_to_disk(Path(output_folder))


if __name__ == "__main__":
    create_dataset()
