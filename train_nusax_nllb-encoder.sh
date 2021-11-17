#!/bin/bash

source $HOME/.bashrc
conda activate nllb-llm2vec

# SEED=${1}
# env HYDRA_FULL_ERROR=1 python -m trident.run experiment=nusax_nllb-encoder_train run.seed=$SEED hydra.run.dir=./logs/nllb-llm2vec/nllb-encoder/nusax/seed-$SEED/
SEED=${1}
env HYDRA_FULL_ERROR=1 python -m trident.run experiment=nusax_nllb-encoder_train run.seed=$SEED hydra.run.dir=./logs/nllb-llm2vec/xlm-r-large/nusax/seed-$SEED/ run.base_model="xlm-roberta-large" run.pretrained_model_name_or_path="xlm-roberta-large"
