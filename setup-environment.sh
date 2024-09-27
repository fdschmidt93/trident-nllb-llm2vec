mamba env create -f ./environment.yaml
conda activate nllb-llm2vec
# does not seem possible yet in conda environment.yaml
pip install flash-attn --no-build-isolation
