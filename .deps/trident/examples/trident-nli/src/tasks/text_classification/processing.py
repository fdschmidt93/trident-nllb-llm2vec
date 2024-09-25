from typing import TypeVar, cast

from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

T = TypeVar("T", None, str, list[str])


# Helper function to handle stripping for both str and list
def _strip_input(input_data: T) -> T:
    if isinstance(input_data, list):
        return cast(T, [instance.strip() for instance in input_data])
    if isinstance(input_data, str):
        return input_data.strip()
    return input_data


def preprocess_fn(
    examples: dict,
    column_names: dict,  # {text, text_pair, labels}
    tokenizer: PreTrainedTokenizerBase,
) -> BatchEncoding:
    """
    Preprocesses text or text pairs using a given tokenizer.

    Args:
        examples (dict): Dictionary containing the input data.
        column_names (dict): Dictionary containing column names for text, text_pair, and labels.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used for preprocessing.

    Returns:
        BatchEncoding: Tokenized representation of the input.

    Extracted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py
    """

    # Extract text and optionally text pairs
    text_column = column_names["text"]
    text = examples[text_column]

    # Check if 'text_pair' exists, and extract if it does
    text_pair_column = column_names.get("text_pair")
    text_pair = examples[text_pair_column] if text_pair_column else None

    text = _strip_input(text)
    text_pair = _strip_input(text_pair)

    # Tokenize the inputs
    result = tokenizer(text=text, text_pair=text_pair)

    return result
