from pathlib import Path
import re
import torch
from trident.utils.logging import get_logger


def get_torch_dtype(type_: str):
    import torch

    return getattr(torch, type_)


def deepsetitem(obj, item, value):
    """
    Sets a value deep in an object based on dot notation.

    If intermediate attributes/dictionaries do not exist, they will be created.

    Does not work if keys comprise dots.

    >>> d = {}
    >>> deepsetitem(d, 'snl_final.about._icsd.icsd_id', 1)
    >>> assert d == {'snl_final': {'about': {'_icsd': {'icsd_id': 1}}}}

    >>> class Example:
    ...     def __init__(self):
    ...         self.data = {}
    ...
    >>> e = Example()
    >>> deepsetitem(e, 'data.snl_final.about._icsd.icsd_id', 2)
    >>> assert e.data == {'snl_final': {'about': {'_icsd': {'icsd_id': 2}}}}
    """

    # Now set the value at the final location
    parts = item.split(".")
    for i in range(len(parts) - 1):
        part = parts[i]
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            # Create new dict if the part is not existing
            new_obj = {}
            if isinstance(obj, dict):
                obj[part] = new_obj
            else:
                setattr(obj, part, new_obj)
            obj = new_obj

    # Set the final part of the path
    if isinstance(obj, dict):
        obj[parts[-1]] = value
    else:
        setattr(obj, parts[-1], value)


def split_by_node_if_multi_gpu(dataset, world_size: int | list[int] = 1):
    num_devices = world_size if isinstance(world_size, int) else len(world_size)
    if num_devices == 1:
        return dataset

    import torch
    from datasets.distributed import split_dataset_by_node
    from trident.utils.logging import get_logger

    log = get_logger(__name__)
    rank = torch.distributed.get_rank()
    dataset_ = split_dataset_by_node(dataset=dataset, world_size=num_devices, rank=rank)
    log.info(f"Rank {rank}: Split dataset by node")
    return dataset_


def get_highest_step_file(directory):
    """
    Returns the path of the subfile with the highest "step" number that can be loaded without errors using `torch.load`.

    Parameters:
    directory (str or Path): The path to the directory containing checkpoint files.

    Returns:
    str or None: The absolute path of the checkpoint file with the highest step number that can be loaded, or None if no valid checkpoints are found.
    """
    # Initialize logging
    log = get_logger(__name__)

    # Regular expression pattern to match files like 'step=250.ckpt'
    pattern = re.compile(r"step=(\d+)\.ckpt")
    parent_path = Path(directory)

    # Check if directory exists
    if not parent_path.exists():
        log.info("No checkpoints yet available to resume from!")
        return None

    # Extract step numbers and sort in descending order
    step_files = []
    for entry in parent_path.iterdir():
        if entry.is_file():
            match = pattern.fullmatch(entry.name)
            if match:
                step = int(match.group(1))
                step_files.append((step, entry))

    # Sort files by step number in descending order
    step_files.sort(reverse=True, key=lambda x: x[0])

    # Attempt to load the highest step checkpoint first
    for step, file_path in step_files:
        try:
            torch.load(file_path, map_location="cpu")
            return str(file_path.absolute())
        except Exception as e:
            log.warning(f"Failed to load checkpoint {file_path} due to error: {e}")

    # If no valid checkpoint was found
    log.info("No valid checkpoints could be loaded.")
    return None


def get_sighup():
    import signal

    return signal.SIGHUP

def return_none(*args, **kwargs):
    return None
