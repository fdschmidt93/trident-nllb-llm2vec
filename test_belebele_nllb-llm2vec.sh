#!/bin/bash

source $HOME/.bashrc
conda activate nllb-llm2vec

# SEED=${1}
# EPOCH=${2}
# env HYDRA_FULL_ERROR=1 python -m trident.run experiment=belebele_nllb-llm2vec_test run.seed=$SEED run.epoch=$EPOCH hydra.run.dir=./logs/nllb-llm2vec/nllb-llm2vec/test/belebele-clf/seed-$SEED/val_epoch-${EPOCH}/
# EPOCH=${1}
# STEPS=${2}
# for SEED in 42 43 44
# do
# env HYDRA_FULL_ERROR=1 python -m trident.run experiment=belebele_nllb-llm2vec_test +run.steps=${STEPS} run.seed=$SEED run.epoch=$EPOCH 'run.checkpoint="${hydra:runtime.cwd}/logs/nllb-llm2vec/nllb-llm2vec/belebele-distill_steps-${run.steps}/seed-${run.seed}/checkpoints/validation-epoch=${run.epoch}.ckpt"' hydra.run.dir=./logs/nllb-llm2vec/nllb-llm2vec/test/belebele-distill_steps-${STEPS}/seed-${SEED}/val_epoch-${EPOCH}/ 'logger.wandb.name="steps=${run.steps}_model=distill-${run.base_model}_epochs=${trainer.max_epochs}_batch-size=${_log_vars.train_batch_size}_accumulate-grad-batches=${trainer.accumulate_grad_batches}_lr=${module.optimizer.lr}_scheduler=${module.scheduler.num_warmup_steps}_seed=${run.seed}_ckpt-epoch=${run.epoch}"' module.nllb_ckpt=null
# done
#
SEED=${1}
EPOCH=${2}
STEPS=${3}
env HYDRA_FULL_ERROR=1 python -m trident.run experiment=belebele_nllb-llm2vec_test +run.steps=${STEPS} run.seed=$SEED run.epoch=$EPOCH 'run.checkpoint="${hydra:runtime.cwd}/logs/nllb-llm2vec/nllb-llm2vec/belebele-distill_steps-${run.steps}/seed-${run.seed}/checkpoints/validation-epoch=${run.epoch}.ckpt"' hydra.run.dir=./logs/nllb-llm2vec/nllb-llm2vec/test/belebele-distill_steps-${STEPS}/seed-${SEED}/val_epoch-${EPOCH}/ 'logger.wandb.name="steps=${run.steps}_model=distill-${run.base_model}_epochs=${trainer.max_epochs}_batch-size=${_log_vars.train_batch_size}_accumulate-grad-batches=${trainer.accumulate_grad_batches}_lr=${module.optimizer.lr}_scheduler=${module.scheduler.num_warmup_steps}_seed=${run.seed}_ckpt-epoch=${run.epoch}"' module.nllb_ckpt=null
