from typing import Optional, Union

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


def get_val_data(
    from_: Optional[int] = None, to_: Optional[int] = None
) -> torch.Tensor:
    """
    Construct a validation tensor based on the given range.

    Args:
        from_: Starting index of the tensor.
        to_: Ending index of the tensor.

    Returns:
        A tensor filled with zeros, except the range from_ to to_ which is filled with ones.
    """
    examples = None
    if isinstance(from_, int):
        examples = torch.zeros(10 - from_, 10)
        for row, col in enumerate(range(from_, 10)):
            examples[row, col] = 1
    if isinstance(to_, int):
        examples = torch.zeros(to_, 10)
        for row, col in enumerate(range(to_)):
            examples[row, col] = 1

    assert isinstance(examples, torch.Tensor)  # Ensure examples is a tensor
    return examples


class IdentityDataset(Dataset):
    """A simple dataset that returns its input as output."""

    def __init__(
        self, X: Union[torch.Tensor, DictConfig], y: Union[torch.Tensor, DictConfig]
    ) -> None:
        super().__init__()
        self.X: torch.Tensor = (
            X if isinstance(X, torch.Tensor) else hydra.utils.instantiate(X)
        )
        self.y: torch.Tensor = (
            y if isinstance(y, torch.Tensor) else hydra.utils.instantiate(y)
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.X[i], self.y[i])


def collate_fn(
    examples: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        examples: A list of tuples containing examples and labels.

    Returns:
        A batched dictionary of examples and labels.
    """
    X, y = zip(*examples)
    batch = {"examples": torch.vstack(X), "labels": torch.stack(y)}
    return batch
