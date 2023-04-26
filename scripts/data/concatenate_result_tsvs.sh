#!/usr/bin/env bash

set -euo pipefail

folder=${1:-experiments/same-sentence-all-vs-all}
file_name=${2:-metrics_before_after.csv}

xsv cat rows $(find $folder -name ${file_name})
