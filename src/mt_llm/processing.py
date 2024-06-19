from typing import cast
import torch
from datasets.arrow_dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset
from transformers.tokenization_utils_fast import BatchEncoding, PreTrainedTokenizerFast


PADDING_CONST = -999_999


class DataCollatorForSequenceClassification:
    def __init__(
        self,
        llm_tokenizer: PreTrainedTokenizerFast,
        llm_tokenizer_kwargs: dict,
        columns: dict[str, str],
        nllb_tokenizer: None | PreTrainedTokenizerFast = None,
        nllb_tokenizer_kwargs: dict = {},
        llm_eos_token: str = "",
        *args,
        **kwargs,
    ) -> None:
        self.llm_tokenizer = llm_tokenizer
        self.llm_tokenizer_kwargs = llm_tokenizer_kwargs

        if getattr(self.llm_tokenizer, "pad_token_id") is None:
            self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id

        self.nllb_tokenizer = nllb_tokenizer
        self.nllb_tokenizer_kwargs = nllb_tokenizer_kwargs
        self.columns = columns
        self.llm_eos_token = llm_eos_token

    def __call__(self, inputs) -> BatchEncoding:
        text: list[str] = [line[self.columns["text"]] for line in inputs]
        text_pair: None | list[str]
        if "text_pair" in self.columns:
            text_pair = [line[self.columns["text_pair"]] for line in inputs]
        else:
            text_pair = None
        llm_batch = self.llm_tokenizer(
            [t + self.llm_eos_token for t in text],
            [t + self.llm_eos_token for t in text_pair]
            if text_pair is not None
            else None,
            **self.llm_tokenizer_kwargs,
        )
        if self.nllb_tokenizer is not None:
            nllb_batch = self.nllb_tokenizer(
                text, text_pair, **self.nllb_tokenizer_kwargs
            )
            for k, v in nllb_batch.items():
                llm_batch[f"nllb_{k}"] = v

        if "label" in self.columns:
            labels = torch.LongTensor([line[self.columns["label"]] for line in inputs])
            llm_batch["labels"] = labels
        if "sequence_embeds" in inputs[0]:
            llm_batch["sequence_embeds"] = torch.stack(
                [input_["sequence_embeds"] for input_ in inputs], dim=0
            )
        return llm_batch


class DataCollatorForAdaptation:
    def __init__(
        self,
        llm_tokenizer: PreTrainedTokenizerFast,
        nllb_tokenizer: None | PreTrainedTokenizerFast,
        *args,
        **kwargs,
    ) -> None:
        self.llm_tokenizer = llm_tokenizer

        if getattr(self.llm_tokenizer, "pad_token_id") is None:
            self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id

        self.nllb_tokenizer = nllb_tokenizer

    def __call__(self, inputs) -> BatchEncoding:
        llama_inputs = [
            {"input_ids": line["input_ids"], "attention_mask": line["attention_mask"]}
            for line in inputs
        ]
        nllb_inputs = [
            {
                "input_ids": line["nllb_input_ids"],
                "attention_mask": line["nllb_attention_mask"],
            }
            for line in inputs
        ]
        llama_batch = self.llm_tokenizer.pad(
            llama_inputs, return_tensors="pt", padding="max_length", max_length=512
        )
        nllb_batch = self.nllb_tokenizer.pad(
            nllb_inputs, return_tensors="pt", padding="max_length", max_length=512
        )
        for k, v in nllb_batch.items():
            llama_batch[f"nllb_{k}"] = v
        return llama_batch


class DataCollatorForAdaptationGPT:
    def __init__(
        self,
        llm_tokenizer: PreTrainedTokenizerFast,
        nllb_tokenizer: PreTrainedTokenizerFast,
        tokenize_kwargs: dict = {},
        *args,
        **kwargs,
    ) -> None:
        self.llm_tokenizer = llm_tokenizer
        self.tokenize_kwargs = tokenize_kwargs

        if getattr(self.llm_tokenizer, "pad_token_id") is None:
            self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id

        self.nllb_tokenizer = nllb_tokenizer

    def __call__(self, inputs) -> BatchEncoding:
        text = [line["text"] for line in inputs]
        llama_batch = self.llm_tokenizer(text, **self.tokenize_kwargs)
        nllb_batch = self.nllb_tokenizer(text, **self.tokenize_kwargs)
        for k, v in nllb_batch.items():
            llama_batch[f"nllb_{k}"] = v
        return llama_batch


def preprocess_adaptation(
    dataset: Dataset,
    llama_tokenizer: PreTrainedTokenizerFast,
    nllb_tokenizer: PreTrainedTokenizerFast,
    tokenize_kwargs: dict = {
        "truncation": True,
        "return_attention_mask": True,
        "max_length": 512,
    },
    text_column: str = "text",
    num_proc: int = 8,
) -> Dataset:
    """
    Preprocesses a dataset by tokenizing and grouping its text.

    Args:
    - dataset: The input dataset.
    - column_names: Dictionary containing column names.
    - tokenizer: Tokenizer to be used.
    - max_seq_length: Maximum sequence length.
    - num_proc: Number of processes.

    Returns:
    - The preprocessed dataset.
    """
    # Extract and convert column names
    remove_columns = dataset.column_names
    if isinstance(remove_columns, dict):
        remove_columns = list(remove_columns.values())

    def tokenize_dataset(
        examples: dict[str, list[str]],
        llama_tokenizer,
        nllb_tokenizer,
        tokenize_kwargs,
        text_column,
    ):
        llama_batch = llama_tokenizer(examples[text_column], **tokenize_kwargs)
        nllb_batch = nllb_tokenizer(examples[text_column], **tokenize_kwargs)
        out = {}
        for k, v in llama_batch.items():
            out[k] = v
        for k, v in nllb_batch.items():
            out[f"nllb_{k}"] = v
        return out

    dataset = dataset.map(
        function=tokenize_dataset,
        fn_kwargs={
            "llama_tokenizer": llama_tokenizer,
            "nllb_tokenizer": nllb_tokenizer,
            "tokenize_kwargs": tokenize_kwargs,
            "text_column": text_column,
        },
        remove_columns=remove_columns,
        batched=True,
        num_proc=num_proc,
        batch_size=10_000,
    )
    return dataset


class DataCollatorForMC:
    def __init__(self, tokenizer, tokenize_kwargs: dict):
        self.tokenizer = tokenizer
        self.tokenize_kwargs = tokenize_kwargs
        # simplify mean mask
        self.tokenizer.padding_side = "right"

    def __call__(self, inputs: list[dict]):
        batch = self.tokenizer.pad(
            {
                "input_ids": [b["input_ids"] for b in inputs],
                "attention_mask": [b["attention_mask"] for b in inputs],
            },
            **self.tokenize_kwargs,
        )
        mean_mask = [torch.Tensor(b["mean_mask"]) for b in inputs]

        batch["mean_mask"] = pad_sequence(mean_mask, batch_first=True, padding_value=0)
        batch["labels"] = torch.LongTensor([b["labels"] for b in inputs])
        if "choice_embeds" in inputs[0]:
            batch["choice_embeds"] = torch.stack(
                [input_["choice_embeds"] for input_ in inputs], dim=0
            )
        return batch


class EmbeddedDataset(TorchDataset):
    def __init__(self, dataset, tensors, key: str = "choice_embeds") -> None:
        super().__init__()
        self.dataset = dataset
        self.tensors = tensors
        self.key = key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        out = self.dataset[i]
        out[self.key] = self.tensors[i]
        return out


def preprocess_multiple_choice(
    examples: dict,
    tokenizer: PreTrainedTokenizerFast,
    columns: dict[str, str],
    tokenize_kwargs: dict = {
        "max_length": 4096,
    },
    *args,
    **kwargs,
) -> dict:
    choices = list(zip(*[examples[c] for c in columns["choices"]]))
    sanitized_choices = []
    for i, choices_ in enumerate(choices):
        cleaned_choices = []
        for choice in choices_:
            if not isinstance(choice, str) or choice == "":
                cleaned_choices.append("None")
            else:
                cleaned_choices.append(choice)
        sanitized_choice = [
            s.strip()
            for s in tokenizer.batch_decode(
                tokenizer(cleaned_choices, max_length=512, add_special_tokens=False)[
                    "input_ids"
                ]
            )
        ]
        sanitized_choices.append(sanitized_choice)
    choices = sanitized_choices
    text = [
        f"{context} {question}\n1 {choice_[0]}\n2 {choice_[1]}\n3 {choice_[2]}\n4 {choice_[3]}"
        for context, question, choice_ in zip(
            examples[columns["context"]], examples[columns["question"]], choices
        )
    ]
    batch = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        # max_length=tokenize_kwargs["max_length"],
        truncation=False,
        return_offsets_mapping=True,
    )
    N, L = batch["input_ids"].shape
    choice_text = [
        f"1 {choice_[0]}\n2 {choice_[1]}\n3 {choice_[2]}\n4 {choice_[3]}"
        for choice_ in choices
    ]
    choice_length = max([len(ids) for ids in tokenizer(choice_text)["input_ids"]])
    max_context_length = tokenize_kwargs["max_length"] - choice_length
    contexts = examples[columns["context"]]
    questions = examples[columns["question"]]
    while L > tokenize_kwargs["max_length"]:
        if max_context_length > 0:
            context_questions_batch = tokenizer(
                text=contexts,
                text_pair=questions,
                max_length=max_context_length,
                truncation="longest_first",
            )
            context_questions = tokenizer.batch_decode(
                context_questions_batch["input_ids"],
            )
            # Llama tokenizer adds bos token in between text and text_pair as well
            context_questions = [
                cq.replace("<|begin_of_text|>", " ").lstrip()
                for cq in context_questions
            ]
            text = [
                f"{context_question}\n1 {choice_[0]}\n2 {choice_[1]}\n3 {choice_[2]}\n4 {choice_[3]}"
                for context_question, choice_ in zip(context_questions, choices)
            ]
        else:
            text = [
                f"1 {choice_[0]}\n2 {choice_[1]}\n3 {choice_[2]}\n4 {choice_[3]}"
                for choice_ in choices
            ]
        batch = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            # max_length=tokenize_kwargs["max_length"],
            truncation=False,
            return_offsets_mapping=True,
        )
        N, L = batch["input_ids"].shape
        max_context_length -= 500
        max_context_length = max(0, max_context_length)
    offsets = batch["offset_mapping"].tolist()
    spans = []
    mask = torch.zeros((4, N, L), dtype=torch.bool)
    for idx, (t, row, row_choices) in enumerate(zip(text, offsets, choices)):
        N = len(row)
        sub_spans = []
        choice = 3
        string_ = ""
        current_end = None
        current_end_token_idx = None
        # INFO: have to use rstrip because newline token comes from left
        choice_stripped = row_choices[choice].strip().encode("utf-8")
        # we go backwards
        for i in reversed(range(0, N)):
            # skip RHS padding
            if row[i][0] == 0 and row[i][1] == 0:
                continue
            # advance end pointer to left
            if current_end is None:
                current_end = row[i][1]
                # appropriately take inclusive range
                current_end_token_idx = i + 1
            # get current text
            string_ = t[row[i][0] : current_end]
            # strip whitespace for clean comparison
            string_stripped = string_.rstrip().encode("utf-8")
            if (
                string_stripped == choice_stripped
                # # this accounts for string_stripped: _helium and helium
                or choice_stripped in string_stripped
            ):
                # limited lookahead for multibyte chars
                # longest multibyte char is 16
                for k in range(1, 16):
                    j = i - k
                    if j < 0:
                        break
                    string_ = t[row[j][0] : current_end]
                    # strip whitespace for clean comparison
                    string_stripped = string_.strip().encode("utf-8")
                    if choice_stripped.startswith(string_stripped):
                        i = j
                mask[choice, idx, i:current_end_token_idx] = True
                sub_spans.append((row[i][0], current_end))
                choice -= 1
                current_end = None
                string_ = ""
                choice_stripped = row_choices[choice].strip().encode("utf-8")
            # advance pointer to left since string has to end in choice
            elif not string_stripped or not choice_stripped.endswith(string_stripped):
                current_end = None
                current_end_token_idx = None
            if choice == -1:
                break
        sub_spans = list(reversed(sub_spans))
        spans.append(sub_spans)
    for s in spans:
        assert len(s) == 4
    # pivot choice and batch dimension
    mask = mask.permute(1, 2, 0)
    batch["mean_mask"] = mask.float()
    batch.pop("offset_mapping")
    out = {"input_ids": [], "attention_mask": [], "mean_mask": []}
    for row_ids, row_attn_mask, row_mean_mask in zip(
        batch["input_ids"], batch["attention_mask"], batch["mean_mask"]
    ):
        mask_ = row_attn_mask.bool()
        out["input_ids"].append(row_ids[mask_].tolist())
        out["attention_mask"].append(row_attn_mask[mask_].tolist())
        out["mean_mask"].append(row_mean_mask[mask_, :].tolist())
    out["labels"] = list(map(lambda x: int(x) - 1, examples[columns["label"]]))
    out["spans"] = spans
    return out


def preprocess_alignment(
    examples: dict[str, list[str]],
    source_tokenizer: PreTrainedTokenizerFast,
    target_tokenizer: PreTrainedTokenizerFast,
    text_column: str = "text",
    tokenize_kwargs: dict = {
        "max_length": 512,
        "truncation": True,
    },
):
    inputs: list[str] = examples[text_column]
    source_batch = cast(BatchEncoding, source_tokenizer(inputs, **tokenize_kwargs))
    target_batch = cast(BatchEncoding, target_tokenizer(inputs, **tokenize_kwargs))

    batch_source_token_spans = []
    batch_target_token_spans = []
    batch_source_tokens = []
    batch_target_tokens = []
    batch_source_masks = []
    batch_target_masks = []
    out = {}
    for i in range(len(inputs)):
        # collect word to chars
        # end is exclusive!
        word_to_char_spans = []
        j = 0
        # catch HF bug, raises TypeError when sequence is finished
        while True:
            try:
                word_span = source_batch.word_to_chars(i, j)
                word_to_char_spans.append((word_span.start, word_span.end - 1))
                j += 1
            except TypeError as _:
                # only then we correctly caught TypeError sinces sequence is exhausted
                assert source_batch.word_to_tokens(i, j) is None
                break
        # collect chars to tokens
        # TODO: what if they don't cover the same words
        source_token_spans = []
        target_token_spans = []
        for span_ in word_to_char_spans:
            start, end_ = span_
            start_source_token = source_batch.char_to_token(i, start)
            end_source_token = source_batch.char_to_token(i, end_)
            start_target_token = target_batch.char_to_token(i, start)
            end_target_token = target_batch.char_to_token(i, end_)
            # ensure both are found
            if (
                start_source_token
                and end_source_token
                and start_target_token
                and end_target_token
            ):
                source_token_spans.append(
                    torch.LongTensor(range(start_source_token, end_source_token + 1))
                )
                target_token_spans.append(
                    torch.LongTensor(range(start_target_token, end_target_token + 1))
                )
        batch_source_token_spans.append(
            pad_sequence(
                source_token_spans, batch_first=True, padding_value=PADDING_CONST
            )
        )
        batch_target_token_spans.append(
            pad_sequence(
                target_token_spans, batch_first=True, padding_value=PADDING_CONST
            )
        )
        batch_source_tokens.append(source_batch[i].tokens)
        batch_target_tokens.append(target_batch[i].tokens)
    out["input_ids"] = source_batch["input_ids"]
    out["attention_mask"] = source_batch["attention_mask"]
    out["spans"] = batch_source_token_spans
    out["tokens"] = batch_source_tokens
    out["nllb_input_ids"] = target_batch["input_ids"]
    out["nllb_attention_mask"] = target_batch["attention_mask"]
    out["nllb_spans"] = batch_target_token_spans
    out["nllb_tokens"] = batch_target_tokens
    return out


class DataCollatorForTokenAlignedDistillation:
    def __init__(
        self,
        llm_tokenizer,
        nllb_tokenizer,
        tokenize_kwargs: dict = {
            "max_length": 512,
            "padding": "max_length",
            "return_tensors": "pt",
        },
        only_overlapping_tokens: bool = True,
        *args,
        **kwargs,
    ) -> None:
        self.llm_tokenizer = llm_tokenizer
        self.nllb_tokenizer = nllb_tokenizer
        self.tokenize_kwargs = tokenize_kwargs
        if getattr(self.llm_tokenizer, "pad_token_id") is None:
            self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id
        self.only_overlapping_tokens = only_overlapping_tokens

    @staticmethod
    def stack_and_pad_tensors(
        tensor_list: list[torch.Tensor],
        L: None | int = None,
        fill_value: int = PADDING_CONST,
    ) -> torch.Tensor:
        # Determine the maximum dimensions K and L
        tensor_list = [
            tensor if tensor.ndim == 2 else tensor.unsqueeze(1)
            for tensor in tensor_list
        ]
        K = max(tensor.shape[0] for tensor in tensor_list)
        if L is None:
            L = max(tensor.shape[1] for tensor in tensor_list)

        # Number of tensors
        N = len(tensor_list)

        # Create the padded tensor with shape (N, K, L)
        padded_tensor = torch.full((N, K, L), fill_value=fill_value)

        # Copy each tensor into the appropriate slice of the padded tensor
        for i, tensor in enumerate(tensor_list):
            k, l_ = tensor.size()
            padded_tensor[i, :k, :l_] = tensor

        return padded_tensor

    def __call__(self, examples: list[dict], *args, **kwds) -> dict:
        llm_inputs = [
            {
                "input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"],
            }
            for example in examples
        ]
        nllb_inputs = [
            {
                "input_ids": example["nllb_input_ids"],
                "attention_mask": example["nllb_attention_mask"],
            }
            for example in examples
        ]

        llm_batch = self.llm_tokenizer.pad(llm_inputs, **self.tokenize_kwargs)
        nllb_batch = self.nllb_tokenizer.pad(nllb_inputs, **self.tokenize_kwargs)

        llm_N, llm_L = llm_batch["input_ids"].shape
        nllb_N, nllb_L = nllb_batch["input_ids"].shape
        # we want to use embedding bag for which we will have to reshape (N, L, D) tensors to (N * L, D)
        # we then aggregate the spans in the (N * L, D) with torch.nn.EmbeddingBag for which we need the adjusted indices
        llm_offsets = torch.arange(0, llm_N * llm_L, llm_L)
        nllb_offsets = torch.arange(0, nllb_N * nllb_L, nllb_L)
        nllb_mask = self.stack_and_pad_tensors(
            # [torch.LongTensor(example["nllb_span_mask"]) for example in examples]
            [torch.LongTensor(example["nllb_spans"]) for example in examples]
        )
        # non relevant indices are first set to -100_000
        # we set everything irrelevant to 0, because for EmbeddingBag
        # all indices have to be relevant
        # we abuse the fact that every sequence is prepended by a BOS token
        # i.e., the first embedding in the eventual (N, L, D) embeddings will be an irrelevant BOS token
        nllb_mask = nllb_mask + nllb_offsets[:, None, None]
        nllb_mask = torch.where(nllb_mask <= 0, 0, nllb_mask)
        nllb_mask = nllb_mask.view(-1, nllb_mask.shape[-1])
        nllb_mask = nllb_mask[nllb_mask.sum(1) > 0]

        llm_mask = self.stack_and_pad_tensors(
            # [torch.LongTensor(example["span_mask"]) for example in examples]
            [torch.LongTensor(example["spans"]) for example in examples]
        )
        llm_mask = llm_mask + llm_offsets[:, None, None]
        llm_mask = torch.where(llm_mask <= 0, 0, llm_mask)
        llm_mask = llm_mask.view(-1, llm_mask.shape[-1])
        llm_mask = llm_mask[llm_mask.sum(1) > 0]

        # don't average potentially noisy spans just overlapping tokens
        # overlapping tokens typically make up of 60-80% of tokens
        # between NLLB and Llama 3 tokenizers
        if self.only_overlapping_tokens:
            nllb_direct = (nllb_mask > 0).sum(1) == 1
            llm_direct = (llm_mask > 0).sum(1) == 1
            direct_mask = torch.logical_and(nllb_direct, llm_direct)
            nllb_mask = nllb_mask[direct_mask, :]
            nllb_mask = nllb_mask[:, :1]
            llm_mask = llm_mask[direct_mask, :]
            llm_mask = llm_mask[:, :1]

        for k, v in nllb_batch.items():
            llm_batch[f"nllb_{k}"] = v
        llm_batch["bag_ids"] = llm_mask  # .swapaxes(1, 2)
        llm_batch["nllb_bag_ids"] = nllb_mask  # .swapaxes(1, 2)
        return llm_batch
