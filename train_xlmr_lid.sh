#!/usr/bin/env bash

# Non-zeroshot experiment where we finetune XLM-R 
# on all available languages (covered by both corpora)
# using NTREX and then test using FLORES.

num_gpus=$(nvidia-smi -L | wc -l)

python click_finetune_xlmr.py \
	--flores-path ./data-bin/flores-dev-no-orth/ \
	--ntrex-path ./data-bin/ntrex-no-orth/ \
	--model-name "xlm-roberta-base" \
	--finetune-langs 'deu,eng,ukr,est,nob,tur' \
	--test-langs 'nob,eng,kaz,est,deu,fin,swe,ukr' \
	--num-train-epochs 6 \
	--batch-size 100 \
    --logging-steps 100 \
    --warmup-steps 500 \
    --eval-steps -1 \
    --num-gpus "${num_gpus}"
