from typing import Optional

import torch


def bert_score_2d(
    x_embeds: torch.Tensor,
    x_attn_mask: torch.Tensor,
    y_embeds: torch.Tensor,
    y_attn_mask: torch.Tensor,
) -> torch.Tensor:
    # l2 normalization
    # some entries have -100??
    x: torch.Tensor = x_embeds[
        x_attn_mask.clamp(min=0.0, max=1.0).bool()
    ]  # (i, j) or (j,)
    y: torch.Tensor = y_embeds[
        y_attn_mask.clamp(min=0.0, max=1.0).bool()
    ]  # (i, k) or (k,)

    x_ = x / torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    y_ = y / torch.linalg.norm(y, ord=2, dim=-1, keepdim=True)

    # i sentences; j padded tokens; d dimensionality
    if x_.dim == 3:
        scores = torch.einsum("ijd, ikd->ijk", x_, y_)
    else:
        scores = torch.einsum("jd, kd->jk", x_, y_)
    r = scores.max(-2)[0].mean()  # (k,)
    p = scores.max(-1)[0].mean()  # (j,)
    f = 2 * r * p / (r + p)
    return f


def bert_score(
    x_embeds: torch.Tensor,
    x_attn_mask: torch.Tensor,
    y_embeds: torch.Tensor,
    y_attn_mask: torch.Tensor,
) -> torch.Tensor:
    X_N, X_B = x_attn_mask.shape
    Y_N, Y_B = y_attn_mask.shape
    x_ = x_embeds / x_embeds.norm(p=2, dim=-1, keepdim=True)
    y_ = y_embeds / y_embeds.norm(p=2, dim=-1, keepdim=True)

    scores = torch.einsum("abc, ijc->aibj", x_, y_)
    mask = torch.einsum(
        "ab, ij->aibj",
        x_attn_mask,
        y_attn_mask,
    ).bool()
    scores = scores.masked_fill(~mask, -1)
    x_score = (scores.max(-1)[0] * x_attn_mask.view(X_N, 1, X_B)).sum(
        -1
    ) / x_attn_mask.sum(-1)[:, None]
    y_score = (scores.max(-2)[0] * y_attn_mask.view(1, Y_N, Y_B)).sum(
        -1
    ) / y_attn_mask.sum(-1)[None, :]
    bert_score = 2 * x_score * y_score / (x_score + y_score)
    return bert_score


def get_attn_mask(alen: torch.Tensor) -> torch.Tensor:
    N = len(alen)
    max_len = alen.max().item()
    attn_mask = torch.arange(max_len).repeat(N, 1) < alen[:, None]
    return attn_mask


def test_bert_score() -> bool:
    N = 40
    MIN_LEN = 10
    MAX_LEN = 50
    x_seq_len = torch.randint(MIN_LEN, MAX_LEN, (N,))
    x_embeds = torch.randn((N, int(x_seq_len.max().item()), 768))
    x_attn_mask = get_attn_mask(x_seq_len)
    y_seq_len = torch.randint(MIN_LEN, MAX_LEN, (N,))
    y_embeds = torch.randn((N, int(y_seq_len.max().item()), 768))
    y_attn_mask = get_attn_mask(y_seq_len)

    scores = bert_score(x_embeds, x_attn_mask, y_embeds, y_attn_mask)

    for i in range(30):
        for j in range(30):
            x_test = x_embeds[i][x_attn_mask[i]]
            y_test = y_embeds[j][y_attn_mask[j]]
            x_test = x_test / x_test.norm(p=2, dim=-1, keepdim=True)
            y_test = y_test / y_test.norm(p=2, dim=-1, keepdim=True)
            cos_sim = x_test @ y_test.T
            p_ = cos_sim.max(0)[0].mean()
            r_ = cos_sim.max(1)[0].mean()
            score_ = 2 * p_ * r_ / (r_ + p_)
            if not torch.allclose(score_, scores[i, j]):
                return False
    return True


def mrr(scores: torch.Tensor) -> torch.Tensor:
    """Compute MRR from row-aligned matrices of square query-document pairs.

    `mrr` is primarily intended for BLI or sentence-translation retrieval.

    Args:
        preds, torch.Tensor: square matrix of ranking scores with true positives on diagonal

    Returns:
        torch.Tensor: mean reciprocal rank
    """
    N = scores.shape[0]
    rankings = (
        scores.argsort(dim=-1, descending=True)
        == torch.arange(N, device=scores.device)[:, None]
    )
    reciprocal_rank = 1 / (1 + rankings.float().argmax(dim=-1))
    return reciprocal_rank.mean()


def retrieval_acc(scores: torch.Tensor) -> torch.Tensor:
    return (
        (scores.argmax(-1) == torch.arange(scores.shape[0], device=scores.device))
        .float()
        .mean()
    )


def cka(
    x: torch.Tensor, y: Optional[torch.Tensor] = None, reduction: Optional[str] = "mean"
) -> torch.Tensor:
    K = x.dim()
    if y is None:
        N = len(x)
        y = x[N // 2 :]
        x = x[: N // 2]
    assert y.dim() == K
    x_centered = (
        x - x.mean(0, keepdim=True) if not (x.shape[0] == 1 or x.dim() == 1) else x
    )
    y_centered = (
        y - y.mean(0, keepdim=True) if not (y.shape[0] == 1 or y.dim() == 1) else y
    )
    if K == 3:
        x_ = torch.linalg.matrix_norm(
            torch.einsum("abc,abd->acd", x_centered, x_centered),
            ord="fro",
            dim=(-2, -1),
        )
        y_ = torch.linalg.matrix_norm(
            torch.einsum("abc,abd->acd", y_centered, y_centered),
            ord="fro",
            dim=(-2, -1),
        )
        xy = torch.linalg.matrix_norm(
            torch.einsum("abc,abd->acd", x_centered, y_centered),
            ord="fro",
            dim=(-2, -1),
        )
        cka = xy.square() / (x_ * y_)
        if isinstance(reduction, str):
            cka: torch.Tensor = getattr(torch, reduction)(cka)
        return cka
    else:
        D = x_centered.shape[-1]
        D_ = y_centered.shape[-1]
        assert D == D_, "x and y have differing dimensionality!"
        denom = (x_centered @ y_centered.T).norm("fro").square()
        numer = (x_centered.T @ x_centered).norm("fro") * (
            y_centered.T @ y_centered
        ).norm("fro")
        return denom / numer


def mean_bert_score(scores):
    return scores.mean()
