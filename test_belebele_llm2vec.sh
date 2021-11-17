#!/bin/bash

source $HOME/.bashrc
conda activate nllb-llm2vec

SEED=${1}
EPOCH=${2}
env HYDRA_FULL_ERROR=1 python -m trident.run experiment=belebele_llm2vec_test run.seed=$SEED run.epoch=$EPOCH hydra.run.dir=./logs/nllb-llm2vec/llm2vec/test/belebele/seed-$SEED/val_epoch-${EPOCH}/
