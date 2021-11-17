from typing import Any
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from trident.core.module import TridentModule
import torch
from omegaconf.dictconfig import DictConfig
from pathlib import Path
import pandas as pd
from hydra.utils import instantiate


class TranslationModule(TridentModule):
    def __init__(self, generate_kwargs: dict | DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_kwargs = instantiate(generate_kwargs)

    def forward(self, batch: dict[str, torch.Tensor]):
        b = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        return {"preds": self.model.generate(**b, **self.generate_kwargs)}


def decode(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    tokenizer: PreTrainedTokenizerFast,
    text: list[str],
    decode_kwargs: dict = {"skip_special_tokens": True},
    *args,
    **kwargs,
):
    # the translations: list[str] are
    # K: translations per text (i.e., N // len(text))
    # [text1_1, ..., text_1_K, ..., text_k_1, ..., text_k_N] aligned
    translations: list[str] = tokenizer.batch_decode(outputs["preds"], **decode_kwargs)
    N = len(translations)
    assert N % len(text) == 0, "Number of text doesn't align with translations"
    per_text_N = N // len(text)
    for i, k in enumerate(text):
        outputs[k] = translations[i * per_text_N : (i + 1) * per_text_N]
        outputs[f"{k}_source"] = batch[f"{k}_source"]

    return outputs


def store_translations(
    outputs: dict[str, list[str]],
    text: str | list[str],
    others: list[str],
    filename: str,
    dir_: str,
    *args,
    **kwargs,
):
    if isinstance(text, str):
        text = [text]
    dir_path = Path(dir_)
    dir_path.mkdir(parents=True, exist_ok=True)
    filepath = dir_path.joinpath(filename)
    dico = {k: outputs[k] for k in text}
    for k in text:
        dico[f"{k}_source"] = outputs[f"{k}_source"]
    for other in others:
        dico[other] = outputs[other]
    df = pd.DataFrame.from_dict(dico)
    df.to_parquet(str(filepath))


class CollatorForTranslation:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        tokenize_kwargs: dict[str, Any],
        columns: dict[str, list[str]],
    ) -> None:
        self.tokenizer = tokenizer
        self.tokenize_kwargs = tokenize_kwargs
        self.columns = columns

    def __call__(self, inputs: list[dict[str, str]], *args: Any, **kwds: Any) -> Any:
        batch = {}

        texts = []
        for text_column in self.columns["text"]:
            text: list[str] = [line[text_column] for line in inputs]
            batch[f"{text_column}_source"] = text
            texts.extend(text)
        batch_ = self.tokenizer(texts, **self.tokenize_kwargs).data
        for k, v in batch_.items():
            batch[k] = v
        if (others := self.columns.get("others", None)) is not None:
            for column in others:
                batch[column] = [line[column] for line in inputs]
        return batch
