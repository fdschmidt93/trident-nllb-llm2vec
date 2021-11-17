#!/bin/bash

source $HOME/.bashrc
conda activate nllb-llm2vec

SEED=${1}
EPOCH=${2}
STEPS=${3}

if [ "$STEPS" == "null" ]; then
    env HYDRA_FULL_ERROR=1 python -m trident.run experiment=belebele_nllb-llm2vec_distill run.seed=${SEED} run.epoch=${EPOCH} hydra.run.dir=./logs/nllb-llm2vec/nllb-llm2vec/belebele-distill_steps-0/seed-${SEED}/ module.nllb_ckpt=null 
else
    env HYDRA_FULL_ERROR=1 python -m trident.run experiment=belebele_nllb-llm2vec_distill run.seed=${SEED} run.epoch=${EPOCH} hydra.run.dir=./logs/nllb-llm2vec/nllb-llm2vec/belebele-distill_steps-${STEPS}/seed-${SEED}/ +run.steps=${STEPS}
fi

# env HYDRA_FULL_ERROR=1 python -m trident.run experiment=belebele_nllb-llm2vec_distill run.seed=$SEED run.epoch=${EPOCH} hydra.run.dir=./logs/nllb-llm2vec/nllb-llm2vec/belebele-distill/seed-$SEED/ +trainer.log_every_n_steps=10
