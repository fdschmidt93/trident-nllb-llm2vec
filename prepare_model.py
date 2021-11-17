# We first compose the LLM2Vec Model
# so that we can load it later cleanly
# Otherwise, we would have to
# 1. Load Llama
# 2. Apply quantization (we only want this after merging step!)
# 3. Merge in LoRAs

# It is not recommended to post-hoc quantize weights as per PEFT

from transformers import (
    AutoModel,
    AutoConfig,
)
from peft.peft_model import PeftModel
import torch

config = AutoConfig.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    trust_remote_code=True,
)
model = AutoModel.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
model = PeftModel.from_pretrained(
    model,
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
)
model = model.merge_and_unload()  # This can take several minutes on cpu
# Loading unsupervised SimCSE model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + SimCSE (LoRA).
model = PeftModel.from_pretrained(
    model, "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse"
)
model = model.merge_and_unload()  # This can take several minutes on cpu

# HF doesn't wanna save this without this hack
del model._hf_peft_config_loaded
model.save_pretrained("./data/model/llm2vec")
