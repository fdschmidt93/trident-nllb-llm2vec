from typing import Union

from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset


def remap_maskhaner_labels(examples: dict) -> dict:
    """Label indices larger than 6 are remapped to 0."""
    examples["ner_tags"] = [
        [tag if tag < 7 else 0 for tag in instance] for instance in examples["ner_tags"]
    ]
    return examples


def load_masakhaner(*args, **kwargs) -> Union[dict[str, Dataset], Dataset]:
    """Remap MasakhaNER labels to WikiANN.

        Assumes WikiANN and MasakhaNER labels are aligned up to the final two labels.

        - `B-DATE` and `I-DATE` are remapped to 0 (`O`)
        - The feature column of the dataset are augmented to only comprise 2 labels and cut the last two features

        >>    wikiann_labels = {
        >>        "O": 0,
        >>        "B-PER": 1,
        >>        "I-PER": 2,
        >>        "B-ORG": 3,
        >>        "I-ORG": 4,
        >>        "B-LOC": 5,
        >>        "I-LOC": 6,
        >>    }

        >>    masakhaner_labels = {
        >>        "O": 0,
        >>        "B-PER": 1,
        >>        "I-PER": 2,
        >>        "B-ORG": 3,
        >>        "I-ORG": 4,
    _   >>        "B-LOC": 5,
        >>        "I-LOC": 6,
        >>        "B-DATE": 7,
        >>        "I-DATE": 8,
        >>    }

    """
    dataset = load_dataset(*args, **kwargs)
    dataset = dataset.map(remap_maskhaner_labels, batched=True)
    if isinstance(dataset, dict):
        for key, value in dataset.items():
            value.features["ner_tags"].feature.names = value.features[
                "ner_tags"
            ].feature.names[:-2]
            value.features["ner_tags"].feature.num_classes = 7
            dataset[key] = value
    else:
        dataset.features["ner_tags"].feature.names = dataset.features[
            "ner_tags"
        ].feature.names[:-2]
        dataset.features["ner_tags"].feature.num_classes = 7
    return dataset
