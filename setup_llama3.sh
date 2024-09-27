source $HOME/.bashrc
conda activate nllb-llm2vec
env CUDA_HOME=$CONDA_PREFIX python ./prepare_model.py --additional_peft_model "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised" --save_path "./data/model/llm2vec_llama3_supervised/"
env CUDA_HOME=$CONDA_PREFIX python ./prepare_model.py --additional_peft_model "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse" --save_path "./data/model/llm2vec_llama3_unsupervised/"
env CUDA_HOME=$CONDA_PREFIX python ./init_up.py
