from datetime import datetime

from datasets.load import load_metric
from torch import Tensor


def compute_metrics(logits: Tensor, labels: Tensor, label_list: list[str]) -> float:
    """
    Computes the F1 score for the given logits and labels using the provided label list.

    Parameters:
    - logits (torch.Tensor): Model output logits.
    - labels (torch.Tensor): Ground truth labels.
    - label_list (list[str]): List of label names used to decode the labels and logits.
                              This list should be passed explicitly in the Hydra project configuration.

    The Hydra project configuration should resemble:
    ```
    logits: "outputs:logits" # uses trident to get `logits` from `outputs` at end of evaluation epoch
    labels: "outputs:labels" # uses trident to get `labels` from `outputs` at end of evaluation epoch
    label_list:
      - O
      - B-PER
      - I-PER
      - B-ORG
      - I-ORG
      - B-LOC
      - I-LOC
    ```
    Note: A common input for WikiANN would follow the yaml configuration provided above.

    Returns:
    - float: The computed micro F1 score accounting for partially incorrect NER tags per `seqeval`.
    """

    predictions = logits.argmax(dim=-1).cpu().numpy().tolist()
    labels = labels.cpu().numpy().tolist()
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[pred] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[lbl] for (_, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # experiment_id to avoid issues with concurrent runs https://github.com/huggingface/evaluate/issues/382kk
    microtimestamp = str(int(datetime.timestamp(datetime.now()) * 1e6))
    metric = load_metric("seqeval", experiment_id=microtimestamp)
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return results["overall_f1"]
