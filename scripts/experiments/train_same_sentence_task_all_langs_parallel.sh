#!/usr/bin/env bash

SINGLE_EXP_SBATCH_CMD="scripts/experiments/sbatch_train_same_sentence_task.sh"
parallel "SAME_SENTENCE_TASK_LANG={1} sbatch -J {1}-samesent ${SINGLE_EXP_SBATCH_CMD}" :::: ./data/all_languages.txt
