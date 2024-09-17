import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

# Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
tokenizer = AutoTokenizer.from_pretrained("vaibhavad/llama-31-mlm")
config = AutoConfig.from_pretrained("vaibhavad/llama-31-mlm", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "vaibhavad/llama-31-mlm",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
model = PeftModel.from_pretrained(
    model,
    "vaibhavad/llama-31-mlm",
)
model = model.merge_and_unload()  # This can take several minutes on cpu

# Loading unsupervised SimCSE model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + SimCSE (LoRA).
model = PeftModel.from_pretrained(model, "vaibhavad/llama-31-mlm-simcse")

model = model.merge_and_unload()

del model._hf_peft_config_loaded
model.save_pretrained("./data/model/llm2vec-llama3-1")

