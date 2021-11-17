import torch


def get_num_tokens(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], *args, **kwargs
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

    outputs["num_tokens"] = (batch["labels"] != -100).sum().unsqueeze(dim=-1)
    outputs["loss"] = outputs["loss"].unsqueeze(dim=-1)
    return outputs


def compute_perplexity(loss: torch.Tensor, num_tokens: torch.Tensor) -> float:
    """
    Computes perplexity given the loss and number of tokens.

    Args:
    - loss: Loss tensor.
    - num_tokens: Number of tokens tensor.

    Returns:
    - Computed perplexity.
    """
    if loss.numel() == 0 or num_tokens.numel() == 0:
        raise ValueError("Input tensors should not be empty.")

    total_nll = torch.sum(loss * num_tokens.float()) / num_tokens.sum().float()
    perplexity = torch.exp(total_nll).item()
    return perplexity
