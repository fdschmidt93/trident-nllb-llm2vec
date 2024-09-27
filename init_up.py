import hydra
from omegaconf import DictConfig
import os
from typing import cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from src.mt_llm.processing import IterableDataCollatorForTokenAlignedDistillation
from src.llm2vec.modelling_llama import LlamaEncoderModel
from tqdm import tqdm
from src.mt_llm.evaluation import fvu

@hydra.main(config_path=".", config_name="init_up")
def main(cfg: DictConfig):
    N = cfg.N  # Number of tokens to process in total
    BATCH_SIZE = cfg.batch_size
    EMBEDDING_SIZE = cfg.embedding_size  # Assuming LLM hidden size is 4096
    nllb_hidden_size = cfg.nllb_hidden_size  # Assuming NLLB hidden size is 1024
    device = cfg.device  # Device to use

    # Tokenizers
    llm_tokenizer = AutoTokenizer.from_pretrained(
        cfg.llm_base_model, padding_side="right"
    )
    nllb_tokenizer = AutoTokenizer.from_pretrained(cfg.nllb_model_name_or_path)

    # Models
    nllb = AutoModel.from_pretrained(
        cfg.nllb_model_name_or_path,
        _attn_implementation="flash_attention_2",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    ).encoder.eval()

    llm_embeddings = cast(
        nn.Embedding,
        LlamaEncoderModel.from_pretrained(
            cfg.llm_weights,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        ).embed_tokens,
    ).eval()

    # Up-projection layer
    up_proj = nn.Linear(nllb_hidden_size, EMBEDDING_SIZE, bias=False).to(device)

    # Dataset and tokenizer arguments
    dataset = load_dataset(
        cfg.dataset_name,
        name=cfg.dataset_config_name,
        split=cfg.dataset_split,
        streaming=True,
    ).shuffle(0)
    iter_ = iter(dataset)
    tokenize_kwargs = {
        "return_tensors": "pt",
        "return_attention_mask": True,
        "max_length": cfg.max_length,
        "padding": "max_length",
        "truncation": True,
    }

    # Collator class for token alignment
    collator = IterableDataCollatorForTokenAlignedDistillation(
        llm_tokenizer, nllb_tokenizer, tokenize_kwargs
    )

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
                    llm_embeds = llm_embeddings(batch["input_ids"])
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
            lines_collected += batch["input_ids"].shape[0]
            del batch
            torch.cuda.empty_cache()
            pbar.update(llm_embeds.shape[0])

    # Concatenate all accumulated embeddings
    llm_embeds_accum = torch.cat(llm_embeds_accum, dim=0)
    nllb_embeds_accum = torch.cat(nllb_embeds_accum, dim=0)

    # Free GPU memory by deleting models
    llm_embeddings = llm_embeddings.cpu()
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
    up_proj.weight = nn.Parameter(least_squares_solution.t())

    # Evaluation
    llm_embeddings = llm_embeddings.to(device)
    nllb = nllb.to(device)

    examples = [next(iter_) for _ in range(BATCH_SIZE)]
    batch = collator(examples)
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Get LLM and NLLB embeddings
        with torch.inference_mode():
            llm_embeds = llm_embeddings(batch["input_ids"])
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
            # Print loss and FVU
            loss = F.mse_loss(up_proj(nllb_embeds), llm_embeds)
            fvu_ = fvu(llm_embeds, loss)
            print(f"Loss: {loss.item()}, FVU: {fvu_}")

    # Save the up_proj state dict
    output_path = cfg.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(up_proj.state_dict(), output_path)

    print(f"Initialized up_proj successfully and saved to {output_path}")


if __name__ == "__main__":
    main()
