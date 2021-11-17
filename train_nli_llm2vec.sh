#!/bin/bash

source $HOME/.bashrc
conda activate nllb-llm2vec

SEED=${1}
env HYDRA_FULL_ERROR=1 python -m trident.run experiment=nli_llm2vec_train run.seed=$SEED hydra.run.dir=./logs/nllb-llm2vec/llm2vec/nli/seed-$SEED/
