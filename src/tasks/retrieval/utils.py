from datasets.arrow_dataset import Dataset


def concatenate_tatoeba(dataset: Dataset) -> Dataset:
    """Concatenates `source_sentence` and `target_sentence` of tatoeba.

    XTREME Tatoeba of Huggingface datasets typically follow the below schema.

    >> Dataset({
    >>     features: ['source_sentence', 'target_sentence', 'source_lang', 'target_lang'],
    >>     num_rows: 1000
    >> })

    This function Concatenates `source_sentence` and `target_sentence` to

    >> Dataset({
    >>     features: ['sentences'],
    >>     num_rows: 2000
    >> })
    """
    return Dataset.from_dict(
        {
            "sentences": list(dataset["source_sentence"])
            + list(dataset["target_sentence"])
        }
    )


def ckpt_from_task(ckpt: dict) -> dict:
    return {k.replace("roberta.", ""): v for k, v in ckpt.items()}
