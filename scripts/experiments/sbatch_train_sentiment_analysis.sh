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

# Constants
export nusax_dataset_path="./data-bin/nusax_sentiment"
export sentiment_num_epochs=${sentiment_num_epochs:-100}
export sentiment_batch_size=${sentiment_batch_size:-160}
export sentiment_max_length_tokens=${sentiment_max_length_tokens:-128}

function run_exp() {

    local train_lang=$1
    local test_lang=$2

    python model_code/click_sentiment_analysis.py \
        --model-name "xlm-roberta-base" \
        --dataset-path $nusax_dataset_path
        --output-dir "./experiments/sentiment_analysis/${lang}" \
        --logging-dir "./log/sentiment_analysis/${lang}" \
        --num-epochs $sentiment_num_epochs \
        --batch-size $sentiment_batch_size \
        --train-lang $train_lang \
        --test-lang $test_lang \
        --max-length-tokens $sentiment_max_length_tokens \
}

export -f run_exp

run_exp "${SENTIMENT_TRAIN_LANG}" "${SENTIMENT_TEST_LANG}"
