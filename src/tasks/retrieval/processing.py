from typing import Optional, Union

import torch
from omegaconf import ListConfig
from trident.core.module import TridentModule

from src.modules.functional import pooling


def get_hidden_states(batch: dict, *args, **kwargs):
    batch["output_hidden_states"] = True
    return batch


def set_batch_attribute(batch: dict, *args, **kwargs):
    assert kwargs.get("attribute"), "Need to pass attribute"
    batch[kwargs["attribute"]] = True
    return batch


def get_embeds(
    outputs: dict,
    n_layers: Union[int, list] = 1,
    aggregation_type: str = "mean",
    pool_type: Optional[str] = "mean",
    *args,
    **kwargs,
):
    assert "hidden_states" in outputs
    embeds = outputs["hidden_states"]

    if isinstance(n_layers, (list, ListConfig)):
        embeds = [embeds[i] for i in n_layers]
    else:
        embeds = embeds[-n_layers:]

    if n_layers == 1 or (isinstance(n_layers, list) and len(n_layers) == 1):
        embeds = embeds[0]
    else:
        embeds = torch.stack(embeds, dim=-1)
        embeds = getattr(torch, aggregation_type)(embeds, dim=-1)

    attention_mask = None
    if pool_type is not None:
        batch = kwargs.get("batch")
        assert batch is not None
        attention_mask = batch["attention_mask"]

    outputs["embeds"] = (
        getattr(pooling, pool_type)(embeds, attention_mask)
        if pool_type is not None
        else embeds
    )
    return outputs


def cosine_sim(
    outputs: dict[str, torch.Tensor],
    *args,
    **kwargs,
):
    N = outputs["embeds"].shape[0] // 2
    x = outputs["embeds"][:N]
    y = outputs["embeds"][N:]
    x = x / torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    y = y / torch.linalg.norm(y, ord=2, dim=-1, keepdim=True)
    outputs["scores"] = x @ y.T
    return outputs
