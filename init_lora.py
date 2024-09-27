import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from peft import get_peft_model
from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.layer import Linear as LoraLinear

from src.mt_llm.processing import IterableDataCollatorForTokenAlignedDistillation
from src.llm2vec.modelling_llama import LlamaEncoderModel
from src.mt_llm.evaluation import fvu

# =============================================================================
# Constants and Configurations
# =============================================================================

N = 1024  # Number of tokens to process in total
BATCH_SIZE = 8  # Batch size
LORA_RANK = 16
LORA_ALPHA = 32
NLLB_HIDDEN_SIZE = 1024  # NLLB hidden size
LLM_HIDDEN_SIZE = 4096  # LLM hidden size
DEVICE = "cuda:0"  # GPU device

# Scaling factor for LoRA
scaling = LORA_ALPHA / LORA_RANK

# =============================================================================
# Tokenizers
# =============================================================================

llm_tokenizer = AutoTokenizer.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", padding_side="right"
)
nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# =============================================================================
# Models
# =============================================================================

# NLLB Encoder Model
nllb = AutoModel.from_pretrained(
    "facebook/nllb-200-distilled-600M",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
).encoder.eval()

# LLM Model with LoRA
llm = (
    get_peft_model(
        LlamaEncoderModel.from_pretrained(
            "./data/model/llm2vec-llama3-1",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        ),
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules="all-linear",
            lora_dropout=0,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        ),
    )
    .eval()
    .to(DEVICE)
)

# Up-projection layer to match NLLB and LLM dimensions
up_proj = nn.Linear(NLLB_HIDDEN_SIZE, LLM_HIDDEN_SIZE, bias=False).to(DEVICE)
state_dict = torch.load("./data/model/up_proj.pth", map_location=DEVICE)
up_proj.load_state_dict(state_dict, strict=True)

# =============================================================================
# Data Preparation
# =============================================================================

# Tokenization arguments
tokenize_kwargs = {
    "return_tensors": "pt",
    "return_attention_mask": True,
    "max_length": 512,
    "padding": "max_length",
    "truncation": True,
}

# Collator for token alignment
collator = IterableDataCollatorForTokenAlignedDistillation(
    llm_tokenizer, nllb_tokenizer, tokenize_kwargs
)

# Load dataset and create an iterator
dataset = load_dataset(
    "HuggingFaceFW/fineweb", name="default", split="train", streaming=True
)
dataset_iter = iter(dataset)
examples = [next(dataset_iter) for _ in range(BATCH_SIZE)]

# Prepare batch data
batch = collator(examples)
batch = {
    k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
}
llm_N, llm_L = batch["input_ids"].shape
nllb_N, nllb_L = batch["nllb_input_ids"].shape

# =============================================================================
# Initial Inference and Loss Calculation
# =============================================================================

with torch.inference_mode():
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        # LLM outputs without LoRA
        llm.disable_adapter_layers()
        llm_outputs_pre = llm(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        llm.enable_adapter_layers()

        # NLLB outputs projected to LLM space
        nllb_llm_outputs_pre = llm(
            inputs_embeds=up_proj(
                nllb(
                    input_ids=batch["nllb_input_ids"],
                    attention_mask=batch["nllb_attention_mask"],
                ).last_hidden_state
            ),
            attention_mask=batch["nllb_attention_mask"],
        )

        # Aggregate embeddings using embedding bag
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

        # Compute initial loss and FVU
        loss = F.mse_loss(nllb_llm_embeds_pre, llm_embeds_pre).detach().cpu()
        fvu_ = fvu(llm_embeds_pre, loss).detach().cpu()

print(f"Initial loss: {loss.item():.3f} | Initial fvu: {fvu_.item():.3f}")

# =============================================================================
# Collecting LLM Layer Outputs without LoRA
# =============================================================================

llm_outputs = {}
llm_handles = {}
layers = []


def traverse_lora(module, outputs, handles, parent_name=""):
    """
    Recursively traverse the LLM model to find LoRA linear layers
    and register forward hooks to collect their outputs.
    """
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(child, LoraLinear):
            layers.append(full_name)

            # Register forward hook to collect outputs
            def collect_output(_, __, output, hook_name=full_name):
                outputs[hook_name] = output.detach().cpu()

            handles[full_name] = child.register_forward_hook(collect_output)
        else:
            traverse_lora(child, outputs, handles, full_name)


# Traverse LLM model and collect outputs
traverse_lora(llm, llm_outputs, llm_handles)

with torch.inference_mode():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        llm.disable_adapter_layers()
        _ = llm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        llm.enable_adapter_layers()

# Remove hooks after collection
for handle in llm_handles.values():
    handle.remove()
llm_handles.clear()

# =============================================================================
# Collecting Base Layer Outputs from NLLB-LLM
# =============================================================================

nllb_llm_base_layer_outputs = {}
nllb_llm_base_layer_handles = {}


def traverse_base_layer_proj(module, outputs, handles, parent_name=""):
    """
    Recursively traverse the LLM model to find base layers
    and register forward hooks to collect inputs and outputs.
    """
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        if name == "base_layer":
            # Register forward hook to collect inputs and outputs
            def collect_output(_, __, output, hook_name=parent_name):
                outputs[hook_name] = output.detach().cpu()
                handles.pop(hook_name).remove()

            handles[parent_name] = child.register_forward_hook(collect_output)
        else:
            traverse_base_layer_proj(child, outputs, handles, full_name)


# Traverse LLM model and collect base layer outputs
traverse_base_layer_proj(llm, nllb_llm_base_layer_outputs, nllb_llm_base_layer_handles)

# =============================================================================
# Adjusting LoRA B Weights
# =============================================================================

nllb_llm_lora_B_handles = {}


def solve(
    low_rank_embeds: torch.Tensor,
    llm_base_layer_outputs: torch.Tensor,
    nllb_llm_base_layer_outputs: torch.Tensor,
    scaling: float,
):
    """
    Solve for the LoRA B matrix weights to minimize the difference
    between LLM base layer outputs and NLLB-LLM base layer outputs.
    """
    resid = llm_base_layer_outputs - nllb_llm_base_layer_outputs
    lora_B_weight = torch.linalg.lstsq(
        low_rank_embeds.float() * scaling, resid.float()
    ).solution
    return lora_B_weight


def traverse_lora_b_proj(module, handles, parent_name=""):
    """
    Recursively traverse the LLM model to find LoRA B projection layers
    and register forward hooks to adjust their weights.
    """
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        if name == "lora_B":
            # Register forward hook to adjust weights
            def hook(module, inputs, __, hook_name=parent_name):
                handles.pop(hook_name).remove()
                device = inputs[0].device

                # Compute embeddings
                nllb_llm_up_proj_embeds = F.embedding_bag(
                    weight=inputs[0].view(nllb_N * nllb_L, -1),
                    input=batch["nllb_bag_ids"],
                    padding_idx=0,
                )
                nllb_llm_base_layer_inputs = nllb_llm_base_layer_outputs.pop(
                    hook_name
                ).to(device)
                nllb_llm_base_layer_embeds = F.embedding_bag(
                    weight=nllb_llm_base_layer_inputs.view(nllb_N * nllb_L, -1),
                    input=batch["nllb_bag_ids"],
                    padding_idx=0,
                )
                llm_base_layer_inputs = llm_outputs.pop(hook_name).to(device)
                llm_base_layer_embeds = F.embedding_bag(
                    weight=llm_base_layer_inputs.view(llm_N * llm_L, -1),
                    input=batch["bag_ids"],
                    padding_idx=0,
                )

                # Solve for LoRA B weights
                lora_B_weight = solve(
                    nllb_llm_up_proj_embeds,
                    llm_base_layer_embeds,
                    nllb_llm_base_layer_embeds,
                    scaling,
                )

                # Update weights
                module.weight = nn.Parameter(lora_B_weight.T.contiguous())
                torch.cuda.empty_cache()
                out = module(inputs[0])

                # Compute MSE for debugging
                with torch.no_grad():
                    residual = module(nllb_llm_up_proj_embeds)
                    mse_before = F.mse_loss(
                        nllb_llm_base_layer_embeds,
                        llm_base_layer_embeds,
                    )
                    mse_after = F.mse_loss(
                        residual + nllb_llm_base_layer_embeds,
                        llm_base_layer_embeds,
                    )
                print(
                    f"{hook_name}: before {mse_before.item():.4f} | after {mse_after.item():.4f}"
                )
                del (
                    nllb_llm_up_proj_embeds,
                    nllb_llm_base_layer_inputs,
                    nllb_llm_base_layer_embeds,
                    residual,
                    llm_base_layer_inputs,
                    llm_base_layer_embeds,
                    mse_before,
                    mse_after,
                )
                torch.cuda.empty_cache()
                return out

            handles[parent_name] = child.default.register_forward_hook(hook)
        else:
            traverse_lora_b_proj(child, handles, full_name)


# Traverse LLM model and adjust LoRA B projections
traverse_lora_b_proj(llm, nllb_llm_lora_B_handles)

# =============================================================================
# Final Inference and Loss Calculation
# =============================================================================

llm.enable_adapter_layers()
with torch.inference_mode():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # NLLB outputs projected to LLM space after LoRA adjustment
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
        # Compute final loss and FVU
        loss = F.mse_loss(nllb_llm_embeds, llm_embeds_pre).detach().cpu()
        fvu_ = fvu(llm_embeds_pre, loss).detach().cpu()

print(f"Post1 loss: {loss.item():.3f} | Post fvu: {fvu_.item():.3f}")

# =============================================================================
# Verification
# =============================================================================

with torch.inference_mode():
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        # Additional inference for verification
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
print(torch.allclose(nllb_llm_embeds, nllb_llm_embeds_post))  # Should print True
