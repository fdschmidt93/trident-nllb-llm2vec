#!/bin/bash -l
#SBATCH --signal=SIGUSR1@90
#SBATCH --partition=single
#SBATCH --mem=256GB
#SBATCH --gres=gpu:A100:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00

source $HOME/.bashrc
conda activate nllb-llm2vec
srun env HYDRA_FULL_ERROR=1 python -m trident.run experiment=alignment_nllb-llm2vec_llama3-1-unsup trainer.devices=8 run.train_batch_size=4 trainer.accumulate_grad_batches=8 trainer.max_steps=30000 module.optimizer.lr=0.0005 +trainer.log_every_n_steps=1 'hydra.run.dir="${hydra:runtime.cwd}/runs/llama-3-1_unsup/"'
