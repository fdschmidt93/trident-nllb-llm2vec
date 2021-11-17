import torch
from datasets.arrow_dataset import Dataset
import math
from src.tasks.text_classification.evaluation import get_preds


def fvu(x: torch.Tensor, x_hat: torch.Tensor, mse_loss: torch.Tensor):
    """Fraction of Variance Unexplained"""
    d_model = x.shape[-1]
    x = x.view(-1, d_model)
    x_hat = x_hat.view(-1, d_model)

    # compute variance of the original activations
    variance = (x - x.mean(dim=0)).pow(2).mean()

    # return ratio of the MSE to the variance of the original activations
    return mse_loss / variance


def get_num_tokens(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *args,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """
    Updates the outputs dictionary with the number of tokens and reshaped loss.

    Args:
    - outputs: The output dictionary.
    - batch: The input batch dictionary.
    - args, kwargs: Additional arguments (not used in this function but can be passed).

    Returns:
    - Updated outputs dictionary.
    """
    if "labels" not in batch:
        raise ValueError("Expected 'labels' key in batch dictionary.")
    if "loss" not in outputs:
        raise ValueError("Expected 'loss' key in outputs dictionary.")

    # this is the number of tokens HF typically computes CE on
    # labels[:, 1:] is only done in the loss calculation of the forward pass
    outputs["num_tokens"] = (
        (batch["labels"][:, 1:] != -100).sum().unsqueeze(dim=-1).detach().cpu()
    )
    outputs["loss"] = outputs["loss"].unsqueeze(dim=-1).detach().cpu()
    return outputs


def token_weighted_ce(loss: torch.Tensor, num_tokens: torch.Tensor):
    if loss.numel() == 0 or num_tokens.numel() == 0:
        raise ValueError("Input tensors should not be empty.")

    total_nll = torch.sum(loss * num_tokens.float()) / num_tokens.sum().float()
    return total_nll


def compute_perplexity(loss: torch.Tensor, num_tokens: torch.Tensor) -> float:
    """
    Computes perplexity given the loss and number of tokens.

    Args:
    - loss: Loss tensor.
    - num_tokens: Number of tokens tensor.

    Returns:
    - Computed perplexity.
    """
    total_nll = token_weighted_ce(loss, num_tokens)
    perplexity = torch.exp(total_nll).item()
    return perplexity


def bits_per_byte(dataset: Dataset, byte_dataset: Dataset) -> float:
    N_tokens = sum(map(len, dataset["input_ids"]))
    N_bytes = sum(map(len, byte_dataset["input_ids"]))
    return N_tokens / (N_bytes * math.log(2))


def bits_per_token(
    loss: torch.Tensor,
    num_tokens: torch.Tensor,
    bits_per_byte: float,
) -> float:
    total_nll = torch.sum(loss * num_tokens.float()) / num_tokens.sum().float()
    return total_nll.item() * bits_per_byte


def to_cpu(outputs: dict, batch, *args, **kwargs):
    outputs = get_preds(outputs, batch)
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            outputs[k] = v.to("cpu")
    return outputs


def store_embeds(embeds: torch.Tensor, path: str, *args, **kwargs):
    """Stores the outputs of pre-embedding (stage 2) to the path."""
    torch.save(embeds, path)
