#!/bin/bash

source $HOME/.bashrc
conda activate nllb-llm2vec

# EPOCH=${1}
# for SEED in 42 43 44 45 46
# do
#     env HYDRA_FULL_ERROR=1 python -m trident.run experiment=nusax_nllb-llm2vec_test run.seed=$SEED run.epoch=$EPOCH 'run.checkpoint="${hydra:runtime.cwd}/logs/nllb-llm2vec/nllb-llm2vec/nusax-distill/seed-${run.seed}/checkpoints/validation-epoch=${run.epoch}.ckpt"' hydra.run.dir=./logs/nllb-llm2vec/nllb-llm2vec/test/nusax-distill/seed-$SEED/val_epoch-${EPOCH}/ 'logger.wandb.name="model=distill-${run.base_model}_epochs=${trainer.max_epochs}_batch-size=${_log_vars.train_batch_size}_accumulate-grad-batches=${trainer.accumulate_grad_batches}_lr=${module.optimizer.lr}_scheduler=${module.scheduler.num_warmup_steps}_seed=${run.seed}_ckpt-epoch=${run.epoch}"'
# done
SEED=${1}
STEPS=${2}
for EPOCH in 0 1 2 3 4 5 6 7 8 9
do
    env HYDRA_FULL_ERROR=1 python -m trident.run experiment=nusax_nllb-llm2vec_test +run.steps=${STEPS} run.seed=$SEED run.epoch=$EPOCH 'run.checkpoint="${hydra:runtime.cwd}/logs/nllb-llm2vec/nllb-llm2vec/nusax-distill_steps-${run.steps}/seed-${run.seed}/checkpoints/validation-epoch=${run.epoch}.ckpt"' hydra.run.dir=./logs/nllb-llm2vec/nllb-llm2vec/test/nusax-distill_steps-${STEPS}/seed-$SEED/val_epoch-${EPOCH}/ 'logger.wandb.name="steps=${run.steps}_model=distill-${run.base_model}_epochs=${trainer.max_epochs}_batch-size=${_log_vars.train_batch_size}_accumulate-grad-batches=${trainer.accumulate_grad_batches}_lr=${module.optimizer.lr}_scheduler=${module.scheduler.num_warmup_steps}_seed=${run.seed}_ckpt-epoch=${run.epoch}"' module.nllb_ckpt=null
done
