source $HOME/.bashrc
conda activate nllb-llm2vec
env CUDA_HOME=$CONDA_PREFIX python ./prepare_model.py --base_model "vaibhavad/llama-31-mlm" --peft_model "vaibhavad/llama-31-mlm"  --additional_peft_model "vaibhavad/llama-31-mlm-simcse" --save_path "./data/model/llm2vec_llama3-1_unsupervised/"
env CUDA_HOME=$CONDA_PREFIX python ./init_up.py 'llm_weights="${hydra:runtime.cwd}/data/model/llm2vec_llama3-1_unsupervised"' 'output_path="${hydra:runtime.cwd}/data/model/up-proj_llm2vec_llama3-1.pth"'
