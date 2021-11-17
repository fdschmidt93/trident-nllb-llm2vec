import torch


def mean(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    attention_mask_ = attention_mask.clamp(min=0, max=1)
    return (hidden_states * attention_mask_[:, :, None]).sum(1) / attention_mask_.sum(
        -1, keepdim=True
    )


def cls(hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return hidden_states[:, 0, :]


def eos(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    padding_side: str = "right",
    *args,
    **kwargs,
) -> torch.Tensor:
    if padding_side == "right":
        N = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        eos_token_id = attention_mask.sum(1) - 1
        return hidden_states[N, eos_token_id, :]
    else:
        return hidden_states[:, -1, :]
