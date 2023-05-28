#!/usr/bin/env bash

njobs_parallel=$1
corpus=${2:-flores}
mkdir -vp ./scratch/${corpus}-mrr-results

if [ "${corpus}" = "nusax" ]
then
    export all_langs_file=data/all_nusax_languages.txt
else
    export all_langs_file=data/all_languages.txt
fi

parallel --progress --bar --jobs ${njobs_parallel} python model_code/mrr.py --lang1 {1} --lang2 {2} --dataset-to-use ${corpus} --output-file ./scratch/${corpus}-mrr-results/mrr-{1}-{2}.tsv 2>&1 > /dev/null :::: ${all_langs_file} :::: ${all_langs_file}
