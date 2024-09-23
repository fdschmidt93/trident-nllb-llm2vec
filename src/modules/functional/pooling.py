import torch
import torch.nn.functional as F
from typing import cast, Optional


def mean(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        attention_mask_ = attention_mask.clamp(min=0, max=1)
        return (hidden_states * attention_mask_[:, :, None]).sum(
            1
        ) / attention_mask_.sum(-1, keepdim=True)


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
#
#
# def get_padding_offset(attention_mask: torch.Tensor) -> int:
#     """
#     If mask was flattened, give first offset of a padding token.
#     If no padding token exists, return -1
#     """
#     try:
#         return cast(int, (attention_mask.view(-1) == 0).nonzero()[0].item())
#     except IndexError as _:
#         return -1
#
#
# def get_offsets(
#     attention_mask: torch.Tensor, padding_offset: Optional[int] = None
# ) -> torch.Tensor:
#     """
#     [[1 1 1 0 0]
#      [1 1 1 1 1]] becomes
#
#     [[0 1 2 3 3]
#      [5 6 7 8 9]]
#
#     assuming padding_offset 3 was input.
#     """
#     N, L = attention_mask.shape
#     offsets = torch.arange(N * L, device=attention_mask.device).view(N, L)
#     if isinstance(padding_offset, int):
#         offsets[~(attention_mask.bool())] = padding_offset
#     return offsets
#
#
# def mean_embedding_bag(
#     hidden_states: torch.Tensor,
#     offsets: torch.Tensor,
#     padding_idx: int,
#     *args,
#     **kwargs,
# ):
#     token_embeds = hidden_states.view(-1, hidden_states.shape[-1])
#     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#         if padding_idx > -1:
#             return F.embedding_bag(
#                 weight=token_embeds,
#                 input=offsets,
#                 padding_idx=padding_idx,
#             )
#         else:
#             return F.embedding_bag(
#                 weight=token_embeds,
#                 input=offsets,
#             )
#
#
# def get_input_offsets(attention_mask):
#     # Find the indices of non-padded tokens in flattened hidden_states
#     input_indices = attention_mask.view(-1).nonzero(as_tuple=False).squeeze()
#
#     # Compute the offsets: for each sequence, where it starts in the flattened input
#     non_padded_lengths = attention_mask.sum(
#         dim=1
#     )  # Count non-padded tokens per sequence
#     offsets = torch.cat(
#         [
#             torch.tensor([0], device=hidden_states.device),
#             non_padded_lengths.cumsum(dim=0)[:-1],
#         ]
#     )
#     return input_indices, offsets
#
#
# def mean_embedding_bag2(
#     hidden_states: torch.Tensor,
#     input: torch.Tensor,
#     offsets: torch.Tensor,
#     *args,
#     **kwargs,
# ):
#     """
#     Compute the mean of non-padded embeddings using `embedding_bag`,
#     properly handling padding with offsets.
#     """
#     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#         # Flatten hidden_states to 2D: shape (batch_size * seq_len, embedding_dim)
#         _, _, embed_dim = hidden_states.shape
#         token_embeds = hidden_states.view(-1, embed_dim)
#
#         # Use embedding_bag with mode 'mean' and appropriate padding index
#         return F.embedding_bag(
#             input=input,  # Indices of non-padded tokens in flattened form
#             weight=token_embeds,  # The flattened hidden states as embedding matrix
#             offsets=offsets,  # Offsets specifying start of each sequence
#             mode="mean",  # Aggregation mode
#         )
#
#
# def mean_iter(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
#     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#         out = []
#         for hs, mask in zip(hidden_states, attention_mask):
#             out.append(hs[: mask.sum(), :].mean(0))
#         embeds_mean_iter = torch.vstack(out)
#     return embeds_mean_iter
#
#
# N = 256
# L = 512
# hidden_states = torch.randn(N, L, 4096).to("cuda:0")
# attention_mask = (
#     torch.randint(int(L * 0.80), L, (N,))[:, None] >= torch.arange(L)[None]
# ).long()
# attention_mask = attention_mask.to("cuda:0")
# embeds_mean_vec = mean(hidden_states, attention_mask)
# input_, offsets_ = get_input_offsets(attention_mask)
# with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
#     out = []
#     for hs, mask in zip(hidden_states, attention_mask):
#         out.append(hs[: mask.sum(), :].mean(0))
#     embeds_mean_iter = torch.vstack(out)
#     padding_offset = get_padding_offset(attention_mask)
#     offsets = get_offsets(attention_mask, padding_offset)
#     embeds_mb1 = mean_embedding_bag(hidden_states, offsets, padding_offset)
#     embeds_mb2 = mean_embedding_bag2(hidden_states, input_, offsets_)
#
# print(torch.allclose(embeds_mean_vec, embeds_mean_iter))  # true
# print(torch.allclose(embeds_mean_vec, embeds_mb1))  # false
# print(torch.allclose(embeds_mean_vec, embeds_mb2))  # false
# print(torch.allclose(embeds_mean_iter, embeds_mb1))  # false
# print(torch.allclose(embeds_mean_iter, embeds_mb2))  # false
# print(torch.allclose(embeds_mb1, embeds_mb2))  # true
#
# from torch.utils.benchmark import Timer
#
# # Example usage
#
# t_iter = Timer(
#     stmt="mean_iter(hidden_states, attention_mask)",
#     setup="from __main__ import mean_iter",
#     globals={"hidden_states": hidden_states, "attention_mask": attention_mask},
# )
# t_vec = Timer(
#     stmt="mean(hidden_states, attention_mask)",
#     setup="from __main__ import mean",
#     globals={"hidden_states": hidden_states, "attention_mask": attention_mask},
# )
# t_emb2 = Timer(
#     stmt="mean_embedding_bag2(hidden_states, input, offsets)",
#     setup="from __main__ import mean_embedding_bag2",
#     globals={"hidden_states": hidden_states, "input": input_, "offsets": offsets_},
# )
# t_emb1 = Timer(
#     stmt="mean_embedding_bag(hidden_states, offsets, padding_offset)",
#     setup="from __main__ import mean_embedding_bag",
#     globals={
#         "hidden_states": hidden_states,
#         "offsets": offsets,
#         "padding_offset": padding_offset,
#     },
# )
#
# print(t_iter.timeit(1000))
# print(t_vec.timeit(1000))
# # precomputes indices etc
# print(t_emb1.timeit(1000))
# # computes indices etc on the fly
# print(t_emb2.timeit(1000))
