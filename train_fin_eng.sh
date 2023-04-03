#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="0" python click_finetune_mt5.py \
	--flores-path ./data-bin/flores-dev-no-orth/ \
	--ntrex-path ./data-bin/ntrex-no-orth/ \
	--model-name "google/mt5-small" \
	--source-lang fin \
	--target-lang eng \
	--finetune-langs 'fin' \
	--num-train-epochs 20 \
	--batch-size 32 \
    --predict-with-generate
	#--logging-steps 500 \
	#--warmup-steps 500 \
	#--logging-steps 500 \
