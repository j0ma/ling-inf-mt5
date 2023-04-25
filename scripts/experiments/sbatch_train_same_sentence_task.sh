#!/usr/bin/env bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --gres=gpu:V100:2
#SBATCH --mail-user=jonnesaleva@brandeis.edu
#SBATCH --mail-type=END
#SBATCH --output=%x-%j.out

export TRANSFORMERS_CACHE="./scratch/model-cache"
export DATASETS_VERBOSITY=error
export TRANSFORMERS_VERBOSITY=error

function run_exp() {

    local lang=$1

    python model_code/finetune_is_same_sentence.py \
        --output-dir "./experiments/same-sentence-all-vs-all/${lang}" \
        --batch-size 250 \
        --language "${lang}" \
        --num-train-epochs 1 \
        --debug
}

export -f run_exp

run_exp "${SAME_SENTENCE_TASK_LANG}"
