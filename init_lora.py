from peft import get_peft_model
from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.layer import Linear as LoraLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from src.mt_llm.processing import IterableDataCollatorForTokenAlignedDistillation
from src.llm2vec.modelling_llama import LlamaEncoderModel
import os
from tqdm import tqdm
from src.mt_llm.evaluation import fvu

# lora init
# nllb embeds
# normal embeds
# for every linear layer
#   init B such that
#       for nllb' input mse of diff is reduced
# Problem
#   account for intermittent error correction (i.e, sequential init)
# Hook
# llm embed
#   1. get all linear outputs
#   2. release hook
# nllb embed
# Hook 2
#   down proj
#


N = 1024  # Number of tokens to process in total
BATCH_SIZE = 32  # Batch size
EMBEDDING_SIZE = 4096  # Assuming LLM hidden size is 4096
LORA_ALPHA = 16
R = 16
nllb_hidden_size = 1024  # Assuming NLLB hidden size is 1024
device = "cuda:0"  # GPU device

# Tokenizers
llm_tokenizer = AutoTokenizer.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", padding_side="right"
)
nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Models
nllb = AutoModel.from_pretrained(
    "facebook/nllb-200-distilled-600M",
    _attn_implementation="flash_attention_2",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
).encoder.eval()
llm = get_peft_model(
    LlamaEncoderModel.from_pretrained(
        "./data/model/llm2vec-llama3-1",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    ),
    LoraConfig(
        r=R,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",
        lora_dropout=0,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        # init_lora_weights="gaussian",
    ),
).eval()
for p in llm.parameters():
    p.to(device)

scaling = LORA_ALPHA / R


# Up-projection layer
up_proj = nn.Linear(nllb_hidden_size, EMBEDDING_SIZE, bias=False).to(device)
sd = torch.load("./data/model/up_proj.pth", map_location=device)
up_proj.load_state_dict(sd, strict=True)

tokenize_kwargs = {
    "return_tensors": "pt",
    "return_attention_mask": True,
    "max_length": 512,
    "padding": "max_length",
    "truncation": True,
}

# Collator class for token alignment
collator = IterableDataCollatorForTokenAlignedDistillation(
    llm_tokenizer, nllb_tokenizer, tokenize_kwargs
)

# dataset = load_dataset(
#     "HuggingFaceFW/fineweb", name="default", split="train", streaming=True
# )
# iter_ = iter(dataset)
# examples = [next(iter_) for _ in range(BATCH_SIZE)]

test_inputs = [
    {"text": "This is a short batch."},
    {"text": "This is yet another short sequence."},
    {"text": "We keep on adding more sentences."},
    {"text": "Until morale of our batch improves."},
    # {"text": "Here we go again."},
    # {"text": "Germany is next to Poland."},
    # {"text": "Who would have thought?"},
]
batch = collator(test_inputs)
batch = {
    k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
}
llm_N, llm_L = batch["input_ids"].shape
nllb_N, nllb_L = batch["nllb_input_ids"].shape

with torch.inference_mode():
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        llm.disable_adapter_layers()
        llm_outputs_pre = llm(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        llm.enable_adapter_layers()
        nllb_llm_outputs_pre = llm(
            inputs_embeds=up_proj(
                nllb(
                    input_ids=batch["nllb_input_ids"],
                    attention_mask=batch["nllb_attention_mask"],
                ).last_hidden_state
            ),
            attention_mask=batch["nllb_attention_mask"],
        )
        llm_embeds_pre = F.embedding_bag(
            weight=llm_outputs_pre.last_hidden_state.view(llm_N * llm_L, -1),
            input=batch["bag_ids"],
            padding_idx=0,
        )
        nllb_llm_embeds_pre = F.embedding_bag(
            weight=nllb_llm_outputs_pre.last_hidden_state.view(nllb_N * nllb_L, -1),
            input=batch["nllb_bag_ids"],
            padding_idx=0,
        )
        loss = F.mse_loss(nllb_llm_embeds_pre, llm_embeds_pre).detach().cpu()
        fvu_ = fvu(llm_embeds_pre, loss).detach().cpu()

print(f"Initial loss: {loss.item():.3f} | Initial fvu: {fvu_.item():.3f}")

# outputs as if no lora
# outputs if lora: x
llm_outputs = {}
llm_handles = {}
layers = []


def traverse_lora(module, outputs, handles, parent_name=""):
    # Iterate through the named children
    for name, child in module.named_children():
        full_name = parent_name + "." + name if parent_name else name
        if isinstance(child, LoraLinear):
            layers.append(full_name)

            # Register the forward hook to collect outputs
            def collect_output(
                _, inputs, outputs, hook_name=full_name
            ):  # Use default argument to capture full_name
                llm_outputs[hook_name] = outputs.detach()

            handles[full_name] = child.register_forward_hook(collect_output)
        else:
            # Recurse into the child module
            traverse_lora(child, outputs, handles, full_name)


traverse_lora(llm, llm_outputs, llm_handles, "")

with torch.inference_mode():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        llm.disable_adapter_layers()
        _ = llm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        llm.enable_adapter_layers()

for key, handle in llm_handles.items():
    handle.remove()
llm_handles = {}

nllb_llm_base_layer_inputs = {}
nllb_llm_base_layer_outputs = {}
nllb_llm_base_layer_handles = {}


def traverse_base_layer_proj(module, outputs, handles, parent_name=""):
    # Iterate through the named children
    for name, child in module.named_children():
        full_name = parent_name + "." + name if parent_name else name
        if name == "base_layer":
            # Register the forward hook to collect outputs

            def collect_output(
                _, inputs, outputs, hook_name=parent_name
            ):  # Use default argument to capture full_name
                nllb_llm_base_layer_inputs[hook_name] = inputs[0].detach()
                nllb_llm_base_layer_outputs[hook_name] = outputs.detach()
                nllb_llm_base_layer_handles.pop(hook_name).remove()

            nllb_llm_base_layer_handles[parent_name] = child.register_forward_hook(
                collect_output
            )
        else:
            # Recurse into the child module
            traverse_base_layer_proj(child, outputs, handles, full_name)


traverse_base_layer_proj(
    llm, nllb_llm_base_layer_outputs, nllb_llm_base_layer_handles, ""
)

nllb_llm_lora_B_handles = {}

# Lora
# x -> f(x)
# x' -> f'(x)
# x -> low(x) -> up(x) -> f_lora(x)
# x' -> f'(x) + f_lora(x)

# I want
# f(x) == f'(x) + f_lora(x)
# f(x) - f'(x) = f_lora(x)


def solve(
    low_rank_embeds: torch.Tensor,
    llm_base_layer_outputs: torch.Tensor,
    nllb_llm_base_layer_outputs: torch.Tensor,
    scaling: float,
):
    # pseudo_inverse transposed shape of low_rank_embeds
    resid = llm_base_layer_outputs - nllb_llm_base_layer_outputs
    lora_B_weight = torch.linalg.lstsq(low_rank_embeds.float() * scaling, resid.float()).solution
    return lora_B_weight


def traverse_lora_b_proj(module, handles, parent_name=""):
    # Iterate through the named children
    for name, child in module.named_children():
        full_name = parent_name + "." + name if parent_name else name
        if name == "lora_B":
            # Register the forward hook to collect outputs
            def hook(
                module, up_proj_inputs, _, hook_name=parent_name
            ):  # Use default argument to capture full_name
                nllb_llm_lora_B_handles.pop(hook_name).remove()
                nllb_llm_up_proj_embeds = F.embedding_bag(
                    weight=up_proj_inputs[0].view(nllb_N * nllb_L, -1),
                    input=batch["nllb_bag_ids"],
                    padding_idx=0,
                )
                nllb_e = nllb_llm_base_layer_outputs.pop(hook_name)
                nllb_llm_base_layer_embeds = F.embedding_bag(
                    weight=nllb_e.view(nllb_N * nllb_L, -1),
                    input=batch["nllb_bag_ids"],
                    padding_idx=0,
                )
                llm_base_layer_embeds = F.embedding_bag(
                    weight=llm_outputs.pop(hook_name).view(llm_N * llm_L, -1),
                    input=batch["bag_ids"],
                    padding_idx=0,
                )
                lora_B_weight = solve(
                    nllb_llm_up_proj_embeds,
                    llm_base_layer_embeds,
                    nllb_llm_base_layer_embeds,
                    scaling,
                )
                # with torch.no_grad():
                #     module.weight.copy_(lora_B_weight.T)
                # torch.cuda.empty_cache()
                # return module(up_proj_inputs[0])

            nllb_llm_lora_B_handles[parent_name] = child.default.register_forward_hook(
                hook
            )
        else:
            # Recurse into the child module
            traverse_lora_b_proj(child, handles, full_name)


traverse_lora_b_proj(llm, nllb_llm_lora_B_handles, "")

llm.enable_adapter_layers()
with torch.inference_mode():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        nllb_outputs = nllb(
            input_ids=batch["nllb_input_ids"],
            attention_mask=batch["nllb_attention_mask"],
        )
        out = llm(
            inputs_embeds=up_proj(nllb_outputs.last_hidden_state),
            attention_mask=batch["nllb_attention_mask"],
        )
        nllb_llm_embeds = F.embedding_bag(
            weight=out.last_hidden_state.view(nllb_N * nllb_L, -1),
            input=batch["nllb_bag_ids"],
            padding_idx=0,
        )
        loss = F.mse_loss(nllb_llm_embeds, llm_embeds_pre).detach().cpu()
        fvu_ = fvu(llm_embeds_pre, loss).detach().cpu()

print(f"Post1 loss: {loss.item():.3f} | Post fvu: {fvu_.item():.3f}")

with torch.inference_mode():
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        nllb_llm_outputs_post = llm(
            inputs_embeds=up_proj(
                nllb(
                    input_ids=batch["nllb_input_ids"],
                    attention_mask=batch["nllb_attention_mask"],
                ).last_hidden_state
            ),
            attention_mask=batch["nllb_attention_mask"],
        )
        nllb_llm_embeds_post = F.embedding_bag(
            weight=nllb_llm_outputs_post.last_hidden_state.view(nllb_N * nllb_L, -1),
            input=batch["nllb_bag_ids"],
            padding_idx=0,
        )
        loss = F.mse_loss(nllb_llm_embeds_post, llm_embeds_pre).detach().cpu()
        fvu_ = fvu(llm_embeds_pre, loss).detach().cpu()
print(f"Post2 loss: {loss.item():.3f} | Post fvu: {fvu_.item():.3f}")
print(torch.allclose(nllb_llm_embeds, nllb_llm_embeds_post)) # prints false


# Dataset and tokenizer arguments
dataset = load_dataset(
    "HuggingFaceFW/fineweb", name="default", split="train", streaming=True
)
iter_ = iter(dataset)
# Accumulate embeddings
llm_embeds_accum = []
nllb_embeds_accum = []
lines_collected = 0
with tqdm(total=N, desc="Collecting tokens", unit="token") as pbar:
    while lines_collected < N:
        examples = [next(iter_) for _ in range(BATCH_SIZE)]
        batch = collator(examples)
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Get LLM and NLLB embeddings
            with torch.inference_mode():
                llm_embeds = llm(batch["input_ids"])
                nllb_embeds = nllb(
                    input_ids=batch["nllb_input_ids"],
                    attention_mask=batch["nllb_attention_mask"],
                ).last_hidden_state

            # Reshape embeddings for F.embedding_bag and move to CPU
            nllb_N, nllb_L = batch["nllb_input_ids"].shape
            llm_N, llm_L = batch["input_ids"].shape

            nllb_embeds = F.embedding_bag(
                weight=nllb_embeds.view(nllb_N * nllb_L, -1).float(),
                input=batch["nllb_bag_ids"],
                padding_idx=0,
            ).cpu()
            llm_embeds = F.embedding_bag(
                weight=llm_embeds.view(llm_N * llm_L, -1).float(),
                input=batch["bag_ids"],
                padding_idx=0,
            ).cpu()

        llm_embeds_accum.append(llm_embeds)
        nllb_embeds_accum.append(nllb_embeds)

        # Update progress bar with the number of tokens just processed
        lines_collected += batch["input_ids"].shape[
            0
        ]  # Increase by the number of tokens processed
        del batch
        torch.cuda.empty_cache()
        pbar.update(llm_embeds.shape[0])
# Concatenate all accumulated embeddings
llm_embeds_accum = torch.cat(llm_embeds_accum, dim=0)
nllb_embeds_accum = torch.cat(nllb_embeds_accum, dim=0)

# Free GPU memory by deleting models
llm = llm.cpu()
nllb = nllb.cpu()
torch.cuda.empty_cache()

# Move accumulated embeddings back to GPU for computation
llm_embeds_accum = llm_embeds_accum.to(device)
nllb_embeds_accum = nllb_embeds_accum.to(device)

# Compute least squares mapping
with torch.no_grad():
    pseudo_inverse = torch.linalg.pinv(nllb_embeds_accum.float())
    least_squares_solution = torch.mm(pseudo_inverse, llm_embeds_accum.float())

# Set the weight matrix in the linear projection layer
with torch.no_grad():
    up_proj.weight.copy_(
        least_squares_solution.t()
    )  # Transpose to match the shape (4096, 1024)

# eval

llm = llm.to(device)
nllb = nllb.to(device)

examples = [next(iter_) for _ in range(BATCH_SIZE)]
batch = collator(examples)
batch = {
    k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
}

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    # Get LLM and NLLB embeddings
    with torch.inference_mode():
        llm_embeds = llm(batch["input_ids"])
        nllb_embeds = nllb(
            input_ids=batch["nllb_input_ids"],
            attention_mask=batch["nllb_attention_mask"],
        ).last_hidden_state

    # Reshape embeddings for F.embedding_bag and move to CPU
    nllb_N, nllb_L = batch["nllb_input_ids"].shape
    llm_N, llm_L = batch["input_ids"].shape

    nllb_embeds = F.embedding_bag(
        weight=nllb_embeds.view(nllb_N * nllb_L, -1).float(),
        input=batch["nllb_bag_ids"],
        padding_idx=0,
    )
    llm_embeds = F.embedding_bag(
        weight=llm_embeds.view(llm_N * llm_L, -1).float(),
        input=batch["bag_ids"],
        padding_idx=0,
    )
    loss = F.mse_loss(up_proj(nllb_embeds), llm_embeds)
    fvu_ = fvu(llm_embeds, loss)

# Save the up_proj state dict
output_path = "./data/model/up_proj.pth"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save(up_proj.state_dict(), output_path)

print(f"Initialized up_proj successfully and saved to {output_path}")
