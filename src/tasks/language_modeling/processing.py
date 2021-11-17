from itertools import chain
from typing import Callable, Union

from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


def tokenize_function_grouped(
    examples: dict[str, list[str]],
    column_names: dict[str, str],
    tokenizer: Union[Callable, PreTrainedTokenizerBase],
) -> BatchEncoding:
    """Tokenizes text data based on the provided column name and tokenizer."""
    return tokenizer(examples[column_names["text"]])


def group_texts(
    examples: dict[str, list[str]], max_seq_length: int
) -> dict[str, list[list[str]]]:
    """
    Groups texts by concatenating them and splitting them based on max_seq_length.

    Args:
    - examples: The input data containing text.
    - max_seq_length: The maximum length of sequence.

    Returns:
    - A dictionary of grouped texts.
    """
    # Concatenate using chain for efficiency
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # Adjust total_length based on max_seq_length
    total_length = (total_length // max_seq_length) * max_seq_length

    # If after adjustment, total_length is zero, return an empty dictionary
    if total_length == 0:
        return {}

    # Split by chunks of max_seq_length
    return {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }


def preprocess_fn(
    dataset: Dataset,
    column_names: dict[str, str],
    tokenizer: Union[Callable, PreTrainedTokenizerBase],
    max_seq_length: int = 512,
    num_proc: int = 12,
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
        remove_columns = list(remove_columns.values())[0]

    # Tokenize and group texts
    dataset = dataset.map(
        function=tokenize_function_grouped,
        fn_kwargs={"tokenizer": tokenizer, "column_names": column_names},
        remove_columns=remove_columns,
        batched=True,
        num_proc=num_proc,
    )
    dataset = dataset.map(
        function=group_texts,
        fn_kwargs={"max_seq_length": max_seq_length},
        batched=True,
        num_proc=num_proc,
    )
    return dataset


def preprocess_line_by_line(
    examples: dict[str, list[str]], column: str, tokenizer: Callable
) -> BatchEncoding:
    """
    Preprocesses text line by line. Removes empty lines and tokenizes them.

    Args:
    - examples: The input data containing lines of text.
    - column: The column name in the examples containing the lines.
    - tokenizer: Tokenizer to be used.

    Returns:
    - Tokenized text.
    """
    examples[column] = [line for line in examples[column] if line.strip()]
    return tokenizer(text=examples[column])
