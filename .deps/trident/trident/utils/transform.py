from typing import Any
import numpy as np
import torch


def stack_or_pad_2d(tensors: list[torch.Tensor], pad_id: int = -100) -> torch.Tensor:
    """
    Stack along first axis of latter axis is homogenous in length else pad and stack.
    """
    N, D = zip(*[tuple(x.shape) for x in tensors])
    if len(set(D)) != 1:
        out = torch.full_like(
            torch.Tensor(sum(N), max(D)), fill_value=pad_id, device=tensors[0].device
        )
        start = 0
        for t in tensors:
            num, len_ = t.shape
            out[start : start + num, :len_] = t
            start += num
        return out
    return torch.vstack(tensors)


def concatenate_3d(tensors: list[torch.Tensor], pad_id: int = -100) -> torch.Tensor:
    # (N sequences, L individual sequence length, C num classes or D dimension -- typically)
    N, L, D = zip(*[tuple(x.shape) for x in tensors])
    out = torch.full_like(
        torch.Tensor(sum(N), max(L), max(D)),
        fill_value=pad_id,
        device=tensors[0].device,
    )
    start = 0
    for t in tensors:
        num, len_, d = t.shape
        out[start : start + num, :len_, :d] = t
        start += num
    return out


def flatten_dict(inputs: list[dict]) -> dict[str, Any]:
    """Conflates keys of list[dict] and stacks np arrays & tensors along 0-dim."""
    ret: dict[str, Any] = {}
    for input_ in inputs:
        for k, v in input_.items():
            if k not in ret:
                ret[k] = []
            # batch: tuple[torch.Tensor, ...]
            # step_outputs: list[tuple[torch.Tensor]]
            if not isinstance(v, list):
                ret[k].append(v)
            else:
                ret[k].extend(v)
    for k, v in ret.items():
        if isinstance(v[0], torch.Tensor):
            dim = v[0].dim()
            # stack matrices along first axis
            if dim == 2:
                ret[k] = stack_or_pad_2d(v)
            # concatenate vectors
            elif dim == 1:
                ret[k] = torch.cat(v, dim=0)
            elif dim == 0:
                ret[k] = torch.stack(v)
            # pad varying dimension and concatenate
            elif dim == 3:
                ret[k] = concatenate_3d(v)
            else:
                raise NotImplementedError(
                    f"Handling {dim} number of dimensions unimplemented"
                )
        elif isinstance(v[0], np.ndarray):
            ret[k] = np.vstack(v)
        elif isinstance(v[0], tuple) and isinstance(v[0][0], torch.Tensor):
            # [(l1, l2, l3), (l1, l2, l3), ]
            listified = []
            for inner_tuples in v:
                for i, tensor in enumerate(inner_tuples):
                    try:
                        sub_list = listified[i]
                    except IndexError:
                        listified.append([])
                        sub_list = listified[i]
                    sub_list.append(tensor)
            list_ = []
            for tensor_list in listified:
                list_.append(concatenate_3d(tensor_list))
            ret[k] = list_
        else:
            pass
    return ret
