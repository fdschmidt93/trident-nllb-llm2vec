from enum import Enum


class Split(Enum):
    """
    The `Split` enum encapsulates various dataset splits used in Lightning.
    Lightning also refers to `stage` (str) which either is 'fit', 'validate', 'test', or 'predict'.
    'fit' spans 'train' and 'val' splits.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    PREDICT = "predict"
