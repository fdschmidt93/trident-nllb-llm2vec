#!/bin/bash

source $HOME/.bashrc
conda activate nllb-llm2vec

SEED=${1}
env HYDRA_FULL_ERROR=1 python -m trident.run experiment=belebele_nllb-llm2vec_train run.seed=$SEED hydra.run.dir=./logs/nllb-llm2vec/nllb-llm2vec/belebele-clf/seed-$SEED/
