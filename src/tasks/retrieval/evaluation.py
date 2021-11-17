import torch
from datasets.load import load_metric
from trident.core.module import TridentModule
from trident.utils.logging import get_logger

from src.modules.functional.metrics import bert_score
from src.tasks.retrieval.processing import get_embeds

log = get_logger(__name__)


def mrr(scores: torch.Tensor) -> torch.Tensor:
    """Compute MRR from row-aligned matrices of square query-document pairs.

    `mrr` is primarily intended for BLI or sentence-translation retrieval.

    Args:
        scores, torch.Tensor: square matrix of ranking scores

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


def nearest_neighbor(trident_module: TridentModule, outputs: dict, *args, **kwargs):
    # (N, L+P, D)
    trg_embeds = outputs["embeds"]
    trg_labels = outputs["labels"]
    trg_attention_mask = outputs["attention_mask"]
    trg_ids = torch.where(trg_labels != -100)

    trg_embeds = trg_embeds[trg_ids]
    # trg_labels = trg_labels[trg_ids]

    # dataset
    trainer = trident_module.trainer
    dm = trainer.datamodule
    # remove unused columns
    src_dataset = dm._remove_unused_columns(dm.dataset_train["source"])
    from torch.utils.data.dataloader import DataLoader
    from transformers import AutoTokenizer
    from transformers.data.data_collator import DataCollatorForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    collate_fn = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True, max_length=510
    )
    src_dl = DataLoader(src_dataset, 256, shuffle=False, collate_fn=collate_fn)
    src_outputs = trainer.predict(model=trident_module, dataloaders=src_dl)
    # how to get embeds and labels
    src_embeds = []
    src_labels = []
    for batch in src_outputs:
        embeds = get_embeds(trident_module, batch, n_layers=6, pool_type=None)["embeds"]
        # attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        mask = labels != -100
        src_embeds.append(embeds[mask])
        src_labels.append(labels[mask])
    src_embeds = torch.vstack(src_embeds)
    src_labels = torch.cat(src_labels).long()

    mean_ = src_embeds.mean(0, keepdim=True)
    std_ = src_embeds.std(0, keepdim=True)

    trg_embeds = trg_embeds.to("cpu")
    trg_labels = trg_labels.to("cpu")

    # src_embeds = (src_embeds - mean_) / std_
    # trg_embeds = (trg_embeds - mean_) / std_
    trg_embeds = trg_embeds / trg_embeds.norm(2, -1, keepdim=True)
    src_embeds = src_embeds / src_embeds.norm(2, -1, keepdim=True)

    scores = trg_embeds @ src_embeds.T
    outputs["nn_1"] = top_knn(
        trident_module, scores, 1, src_labels, trg_labels, trg_ids
    )
    outputs["nn_3"] = top_knn(
        trident_module, scores, 3, src_labels, trg_labels, trg_ids
    )
    outputs["nn_10"] = top_knn(
        trident_module, scores, 10, src_labels, trg_labels, trg_ids
    )
    outputs["nn_25"] = top_knn(
        trident_module, scores, 25, src_labels, trg_labels, trg_ids
    )
    outputs["nn_50"] = top_knn(
        trident_module, scores, 50, src_labels, trg_labels, trg_ids
    )
    outputs["nn_100"] = top_knn(
        trident_module, scores, 100, src_labels, trg_labels, trg_ids
    )
    outputs["nn_250"] = top_knn(
        trident_module, scores, 250, src_labels, trg_labels, trg_ids
    )
    outputs["nn_500"] = top_knn(
        trident_module, scores, 500, src_labels, trg_labels, trg_ids
    )
    outputs["nn_1000"] = top_knn(
        trident_module, scores, 1000, src_labels, trg_labels, trg_ids
    )
    return outputs


def top_knn(module, scores, topk, train_labels, test_labels, test_ids):
    topk = min(topk, scores.shape[1])
    idx = scores.topk(topk, 1)[1]  # .flatten()
    pred = train_labels[idx]
    if topk > 1:
        pred = pred.mode(1)[0]
    else:
        pred = pred.flatten()
    # top1_acc = (pred == test_labels).sum() / test_labels.shape[0]
    predictions = torch.full_like(test_labels, -100, device=test_labels.device)
    pred = pred.long()
    predictions = predictions.long()
    predictions[test_ids] = pred
    metric = load_metric("seqeval")
    labels = test_labels.long().detach().cpu().numpy().tolist()
    predictions = predictions.long().detach().cpu().numpy().tolist()
    # Remove ignored index (special tokens)
    true_predictions = [
        [
            module.trainer.datamodule.label_list[p]
            for (p, l) in zip(prediction, label)
            if l != -100
        ]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [
            module.trainer.datamodule.label_list[l]
            for (p, l) in zip(prediction, label)
            if l != -100
        ]
        for prediction, label in zip(predictions, labels)
    ]
    metric = load_metric("seqeval")
    result = metric.compute(predictions=true_predictions, references=true_labels)[
        "overall_f1"
    ]
    log.info(f"Top{topk}-NN: {result:.4f}")
    return result


def pair_wise_loop(
    self,
    outputs: dict[str, torch.Tensor],
    # pair_wise_metric: Callable,
    batch_size: int = 20,
    *args,
    **kwargs,
) -> dict:
    """

    [TODO:description]

    Args:
        inputs, dict[str, torch.Tensor]:
        metric: [TODO:description]
        pair_wise: [TODO:description]

    Returns:
        Union[torch.Tensor, float]: [TODO:description]
    """
    device = outputs["attention_mask"].device
    N = outputs["attention_mask"].shape[0] // 2
    scores = torch.empty((N, N), device=device)
    # scores_ = torch.empty((N, N), device=device)

    x_embeds = outputs["embeds"][:N]
    x_attn_mask = outputs["attention_mask"][:N].clamp(min=0.0, max=1.0)
    y_embeds = outputs["embeds"][N:]
    y_attn_mask = outputs["attention_mask"][N:].clamp(min=0.0, max=1.0)

    chunks = torch.arange(N, dtype=torch.long).reshape(-1, batch_size).to(device)
    for chunk in chunks:
        chunk_scores = bert_score(
            x_embeds[chunk], x_attn_mask[chunk], y_embeds, y_attn_mask
        )
        scores[chunk, :] = chunk_scores
    outputs["scores"] = scores
    return outputs


def mean_cosine_sim(
    embeds: torch.Tensor,
):
    N = embeds.shape[0] // 2
    x = embeds[:N].mean(0)
    y = embeds[N:].mean(0)
    x = x / torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    y = y / torch.linalg.norm(y, ord=2, dim=-1, keepdim=True)
    return x.T @ y
