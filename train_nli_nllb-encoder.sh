#!/bin/bash

source $HOME/.bashrc
conda activate nllb-llm2vec

SEED=${1}
# env HYDRA_FULL_ERROR=1 python -m trident.run experiment=nli_nllb-encoder_train run.seed=$SEED hydra.run.dir=./logs/nllb-llm2vec/nllb-encoder/nli/seed-$SEED/
env HYDRA_FULL_ERROR=1 python -m trident.run experiment=nli_nllb-encoder_train run.seed=$SEED hydra.run.dir=./logs/nllb-llm2vec/xlm-r-base/nli/seed-$SEED/ run.base_model="xlm-roberta-base" run.pretrained_model_name_or_path="xlm-roberta-base" 
