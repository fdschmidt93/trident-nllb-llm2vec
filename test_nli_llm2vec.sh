#!/bin/bash

source $HOME/.bashrc
conda activate nllb-llm2vec

EPOCH=${1}
for SEED in 42 43 44
do
    env HYDRA_FULL_ERROR=1 python -m trident.run experiment=nli_llm2vec_test run.seed=$SEED run.epoch=$EPOCH hydra.run.dir=./logs/nllb-llm2vec/llm2vec/test/nli/seed-$SEED/val_epoch-${EPOCH}/
done
