#!/usr/bin/env python

import pickle
from pathlib import Path
import itertools as it

import click
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
# from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_chunked
from scipy.stats import rankdata

import datasets as ds

def add_sentence_id(examples):
    examples['sentence_id'] = [idx for idx, _ in enumerate(examples['text'])]
    return examples

def mrr_experiment_lang_vs_lang2_pairwise(lang, lang2, embs, df):
    assert embs.shape[0] == df.shape[0]

    # Filter DataFrames by language only once and store them in a dictionary
    lang_dfs = {lang: df[df.language == lang].copy().reset_index(drop=True) for lang in (lang, lang2)}

    embs_lang = embs[lang_dfs[lang].global_sentence_id]
    embs_lang2 = embs[lang_dfs[lang2].global_sentence_id]

    # Use pairwise_distances_chunked to compute pairwise distances in smaller chunks
    distances_iter = pairwise_distances_chunked(embs_lang, embs_lang2, metric="cosine")

    mrr_sum = 0
    count = 0
    for distances_chunk in distances_iter:
        # Compute ranks using rankdata from scipy.stats
        ranks = rankdata(distances_chunk, axis=1) - 1

        sentence_ids = lang_dfs[lang].sentence_id.values
        target_indices = np.where(ranks == sentence_ids[:, np.newaxis])[1]
        mrr_sum += np.sum(1 / (target_indices + 1))
        count += len(target_indices)

    mrr = mrr_sum / count
    return mrr

def all_ranks_dists_lang_vs_lang2_pairwise(lang, lang2, embs, df):
    assert embs.shape[0] == df.shape[0]

    # Filter DataFrames by language only once and store them in a dictionary
    lang_dfs = {lang: df[df.language == lang].copy().reset_index(drop=True) for lang in (lang, lang2)}

    embs_lang = embs[lang_dfs[lang].global_sentence_id]
    embs_lang2 = embs[lang_dfs[lang2].global_sentence_id]

    # Use pairwise_distances_chunked to compute pairwise distances in smaller chunks
    distances_iter = pairwise_distances_chunked(embs_lang, embs_lang2, metric="cosine")

    mrr_sum = 0
    count = 0
    all_ranks = []
    all_distances = []
    for distances_chunk in distances_iter:
        # Compute ranks using rankdata from scipy.stats
        ranks = rankdata(distances_chunk, axis=1)
        all_ranks.append(ranks)
        all_distances.append(distances_chunk)

    if len(all_ranks) == 1:
        all_ranks = all_ranks[0]
    if len(all_distances) == 1:
        all_distances = all_distances[0]
        
    return all_ranks, all_distances

def sim_search_results(lang1, lang2, embs, df, **kwargs):
    
    _ranks, _ = all_ranks_dists_lang_vs_lang2_pairwise(lang1, lang2, embs, df)
    ranks_that_matter = pd.Series(_ranks.diagonal())
    
    out = {
        'avg_rank' : ranks_that_matter.mean(),
        'std_rank': ranks_that_matter.std(),
        'mrr': ranks_that_matter.apply(lambda x: 1/x).mean(),
        'lang1': lang1, 'lang2': lang2
    }
    
    return out


@click.command()
@click.option("--lang1")
@click.option("--lang2")
@click.option("--dataset-to-use")
@click.option("--output-file", default="mrr_results.tsv")
def main(lang1, lang2, dataset_to_use, output_file):

    # LOAD FLORES & NTREX
    flores_separate_langs = ds.load_from_disk("./data-bin/flores-dev-no-orth/")
    ntrex_separate_langs = ds.load_from_disk("./data-bin/ntrex-no-orth//")

    flores_separate_langs = flores_separate_langs.map(add_sentence_id, batched=True, batch_size=10000)
    ntrex_separate_langs = ntrex_separate_langs.map(add_sentence_id, batched=True, batch_size=10000)

    flores = ds.concatenate_datasets([
        flores_separate_langs[lang] for lang in flores_separate_langs
    ])
    ntrex = ds.concatenate_datasets([
        ntrex_separate_langs[lang] for lang in ntrex_separate_langs if lang in flores_separate_langs
    ])

    # LOAD NUSAX
    nusax_paths = {
        split: Path("./data/nusax_parallel_sentences") / f"{split}.csv"
        for split in ["train", "valid", "test"]
    }
    nusax_dataset_dict = ds.DatasetDict()
    for split, p in nusax_paths.items():
        _split = {"dev": "valid"}.get(split, split)    
        _ds = ds.Dataset.from_csv(str(p)).remove_columns('Unnamed: 0')
        nusax_dataset_dict[_split] = _ds

    nusax = ds.concatenate_datasets([nusax_dataset_dict[split] for split in nusax_dataset_dict])
    nusax_df = nusax.to_pandas()
    nusax_df.columns.name = "language"
    nusax_sentences = nusax_df.stack()
    nusax_sentences.index.names = ['sentence_id', 'language']
    nusax_sentences.name = 'text' 
    nusax_df = nusax_sentences.reset_index().sample(frac=1) # randomize order
    nusax = ds.Dataset.from_pandas(nusax_df, preserve_index=False)

    # Combine datasets
    all_datasets = {"flores": flores, "nusax": nusax, "ntrex": ntrex}
    all_dataset_dfs = {
        key: val.to_pandas() for key, val in all_datasets.items()
    }
    for dataset_name, df in all_dataset_dfs.items():
        df['global_sentence_id'] = df.index.tolist()

    print("ranking")

    # DO ACTUAL RANKING
    model_names = { "mbert", "xlmr", "mbert_cls", "xlmr_cls", "mbert_stsb", "xlmr_stsb" }

    # load pickle file
    with open('data/retrieval_featurized_sentences_flores_nusax_ntrex.pkl', 'rb') as f:
        featurized_sentences = pickle.load(f)

    rows_mrr_experiment_analysis_pairwise = []
    for model_name, dataset_name in it.product(model_names, [dataset_to_use]):
        embeddings = featurized_sentences[dataset_name][model_name]
        df = all_dataset_dfs[dataset_name]
        result = sim_search_results(lang1, lang2, embeddings, df)
        result['dataset'] = dataset_name
        result['model'] = model_name

        # mrr = mrr_experiment_lang_vs_lang2_pairwise( lang1, lang2, embeddings, df)
        # rows_mrr_experiment_analysis_pairwise.append(
            # (dataset_name, model_name, lang1, lang2, mrr)
        # )
        rows_mrr_experiment_analysis_pairwise.append(result)

    # mrr_results = pd.DataFrame(rows_mrr_experiment_analysis_pairwise, columns=["dataset", "model", "query_lang", "corpus_lang", "mrr"])
    mrr_results = pd.DataFrame(rows_mrr_experiment_analysis_pairwise)
    mrr_results.to_csv(output_file, index=False, sep="\t")

if __name__ == '__main__':
    main()
