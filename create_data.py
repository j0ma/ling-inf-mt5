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
    "--output-folder",
    type=click.Path(),
    required=True,
    help="Path to the folder where the train and test datasets will be saved.",
)
@click.option(
    "--test-size",
    type=float,
    default=0.2,
    help="Fraction of the data to be used for testing.",
)
@click.option("--flores", is_flag=True)
@click.option("--get-rid-of-orth-flores", is_flag=True)
def create_dataset(
    data_folder, output_folder, test_size, flores=False, get_rid_of_orth_flores=False
):
    # Get a list of all the parallel data files in the data folder
    file_names = list(Path(data_folder).glob("*.dev"))

    # Load the data from each file into a list of dictionaries
    data = ds.DatasetDict()

    def add_lang(d, lang="fin"):
        d["language"] = lang

        return d

    for file_name in file_names:
        with open(file_name, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f]
            lang = file_name.name.replace(".dev", "")

            if flores and get_rid_of_orth_flores:
                lang = lang.split("_")[0]
            data[lang] = ds.Dataset.from_dict({"text": sentences}).map(
                functools.partial(add_lang, lang=lang)
            )

    # Save the train and test datasets to disk
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    data.save_to_disk(Path(output_folder))


if __name__ == "__main__":
    create_dataset()
