#!/usr/bin/env bash

set -euo pipefail

function run_single_sbatch () {
    local lang=${1}
    export SAME_SENTENCE_TASK_LANG=${lang}
    job_name=${lang}-samesent

    sbatch \
        -J "${job_name}" \
        scripts/experiments/sbatch_train_same_sentence_task.sh
}

export -f run_single_sbatch

parallel "run_single_sbatch {1}" :::: ./data/all_languages.txt
