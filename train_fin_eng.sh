#!/usr/bin/env bash

num_gpus=$(nvidia-smi -L | wc -l)

python -m pudb click_finetune_mt5.py \
	--flores-path ./data-bin/flores-dev-no-orth/ \
	--ntrex-path ./data-bin/ntrex-no-orth/ \
	--model-name "google/mt5-small" \
	--source-lang fin \
	--target-lang eng \
	--finetune-langs 'fin' \
	--num-train-epochs 600 \
	--batch-size 20 \
    --predict-with-generate \
    --logging-steps 600 \
    --warmup-steps 600 \
    --logging-steps 600 \
    --finetune-all-langs-except-src-tgt \
    --num-gpus "${num_gpus}"
