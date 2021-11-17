Disclaimer: The code still undergoes refactoring to be much more usable and clean out-of-the-box.

# NLLB-LLM2Vec

## Setup & Installation

### Cloning repositories

You can install required pinned dependencies with the below command.

### Setting up environment

For now, [miniconda](https://github.com/conda-forge/miniforge) is the recommended tool to manage dependencies for this project.
```bash
cd ./trident-nllb-llm2vec/
conda env create -f environment.yaml
```

### Initial setup of LLM2Vec

On a machine with a Nvidia-GPU, run

```bash
cd ./trident-nllb-llm2vec/
python ./prepare_model.py
```

This downloads Llama 3 8B, LLM2Vec adapters, merges the required adapters, and stores the model into the appropriate folder.


# General adaptation

The NLLB-LLM2Vec adaptation requires downloading the FineWeb 10BT dataset with the following script.

```bash
cd ./trident-nllb-llm2vec/
bash ./download_fineweb.sh
```

Then the model can be trained. We trained the model on 8 A100 80GB for 10K steps (~22 hours) with the below script.

```bash
python -m trident.run experiment=adaptation_nllb-llm2vec.yaml hydra.run.dir=$OUTPUT_FOLDER

```

You must set the output folder where checkpoints get stored to.

# Task Fine-tuning

You can train LLM2Vec on a particular `$TASK` as follows.

```bash
cd ./trident-nllb-llm2vec/
bash train_llm2vec_$TASK.sh $SEED
```

We ran with seeds 42, 43, 44 (NLI & Belebele) and 42, 43, 44, 45, 46 for NusaX.

The outputs are then stored to `./trident-nllb-llm2vec/logs/nllb-llm2vec/llm2vec/nli/seed-$SEED/`

# Task Distillation

Task distillation first requires to pre-embed the training datasets with the fine-tuned LLM2Vec models which takes 30-60 minutes depending on your GPU infrastructure.
You need to check what checkpoint (`$EPOCH`) performed best on source-language validation instances on `wandb`.

```bash
cd ./trident-nllb-llm2vec/
bash preembed_llm2vec_$TASK.sh $SEED $EPOCH
```

Then you can run

```bash
cd ./trident-nllb-llm2vec/
bash distill_nllb-llm2vec_$TASK.sh $SEED $EPOCH
```

At last, you can evaluate your model as follows.


```bash
cd ./trident-nllb-llm2vec/
# evaluates on all 3 (or 5, for NusaX) seeds already
bash test_nllb-llm2vec_$TASK.sh $EPOCH
```


Evaluation can be very costly due to the number languages that are being evaluated. Belebele requires about 3hr on a A100 40GB.

