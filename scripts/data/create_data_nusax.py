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
def create_dataset(
    data_folder,
    output_folder,
):
    def add_lang(d, lang="fin"):
        d["language"] = lang

        return d

    # data = ds.DatasetDict()
    from collections import defaultdict
    datas = defaultdict(list)

    language_subfolders = list(p for p in Path(data_folder).glob("*") if p.is_dir())
    from functools import partial

    for lang_sf in language_subfolders:
        language = lang_sf.name
        csv_files = list(str(p) for p in lang_sf.glob("*.csv"))
        _add_lang = partial(add_lang, lang=language)
        lang_dataset = ds.load_dataset(
            "csv", data_files=dict(zip(["train", "valid", "test"], csv_files))
        ).map(_add_lang)

        for split in ["train", "valid", "test"]:
            datas[split].append(lang_dataset[split])

    data = ds.DatasetDict()
    data['train'] = ds.concatenate_datasets(datas['train'])
    data['valid'] = ds.concatenate_datasets(datas['valid'])
    data['test'] = ds.concatenate_datasets(datas['test'])

    # Save the train and test datasets to disk
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    data.save_to_disk(Path(output_folder))


if __name__ == "__main__":
    create_dataset()
