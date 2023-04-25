#!/usr/bin/env bash

set -euo pipefail

folder=${1:-experiments/same-sentence-all-vs-all}
file_name=${2:-metrics_before_after.csv}

find $folder -name ${file_name} \
    | xargs -I {} xsv cat rows {}
