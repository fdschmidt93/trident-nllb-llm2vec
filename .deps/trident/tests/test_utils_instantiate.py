import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from trident.core.dataspec import TridentDataspec


def tensor_method_apply_helper(s: str):
    cfg = OmegaConf.create(s)
    assert isinstance(cfg, DictConfig)
    dataspec = TridentDataspec(cfg)
    ref_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    ref_tensor = ref_tensor + 1
    ref_tensor = ref_tensor * 2
    assert torch.sum(dataspec.dataset).item() == torch.sum(ref_tensor).item()


def tensor_method_apply_inplace_helper(s: str):
    cfg = OmegaConf.create(s)
    assert isinstance(cfg, DictConfig)
    dataspec = TridentDataspec(cfg)
    # check that "rename" is equivalent to permuting tensor columns (if needed)
    ref_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    ref_tensor = ref_tensor + 1
    ref_tensor = ref_tensor * 2
    assert torch.sum(dataspec.dataset).item() == torch.sum(ref_tensor).item()


def test_method():
    s = """
    dataset:
      _target_: torch.Tensor
      _convert_: all
      data:
        - [1, 2]
        - [3, 4]
    preprocessing:
      method:
        add:
          other: 1
        mul:
          other: 2
    """
    tensor_method_apply_helper(s)


def test_method_inplace():
    s = """
    dataset:
      _target_: torch.tensor
      _convert_: all
      data:
        - [1, 2]
        - [3, 4]
    preprocessing:
      method:
        add_:
          other: 1
        mul_:
          other: 2
    """
    tensor_method_apply_inplace_helper(s)


def test_apply():
    s = """
    dataset:
      _target_: torch.tensor
      _convert_: all
      data:
        - [1, 2]
        - [3, 4]
    preprocessing:
      apply:
        add:
          _target_: torch.add
          other: 1
        mul:
          _target_: torch.mul
          other: 2
    """
    tensor_method_apply_helper(s)


def test_apply_inplace():
    s = """
    dataset:
      _target_: torch.tensor
      _convert_: all
      data:
        - [1, 2]
        - [3, 4]
    preprocessing:
      apply:
        add:
          _target_: torch.Tensor.add_
          other: 1
        mul:
          _target_: torch.Tensor.mul_
          other: 2
    """
    tensor_method_apply_inplace_helper(s)
