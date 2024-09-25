import torch
import torch.nn.functional as F


def get_preds(outputs: dict, *args, **kwargs) -> dict:
    r"""
    Computes predictions based on the logits in the given outputs dictionary and appends them to the dictionary.

    This is a reference implementation showing how one can extract predictions from the logits
    produced by a model. The function updates the provided `outputs` dictionary with a new key, "preds".

    Args:
        outputs (dict): Dictionary containing model outputs, especially "logits".
        *args: Variable length argument list.
        **kwargs: keyword arguments can include:
            - `trident_module` (:obj:`TridentModule`): An instance of the module.
            - `batch` (dict): Batch of input data.
            - `split` (:obj:`trident.utils.enums.Split`): Data split type (e.g., train, test, validation).

    Return:
        dict: Updated dictionary with the computed predictions added as "preds".

    .. code-block:: python

        outputs = {"logits": torch.tensor([[2.0, 1.0], [1.0, 2.0]])}
        updated_outputs = get_preds(outputs)
        print(updated_outputs["preds"])  # Outputs: tensor([0, 1])

    Examples::

        outputs = {"logits": torch.tensor([[2.0, 1.0], [1.0, 2.0]])}
        updated_outputs = get_preds(outputs)
        assert updated_outputs["preds"].tolist() == [0, 1]

    Note:
        This function serves as a reference for users wanting to implement prediction extraction
        for their custom models. Depending on the model architecture and desired behavior, the
        implementation might vary.

    """
    outputs["preds"] = outputs["logits"].argmax(-1)
    return outputs


def get_loss(preds: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    N = logits.shape[0]
    device = logits.device
    exp_loss = -F.log_softmax(logits, -1)[torch.arange(N, device=device), preds].mean()
    return exp_loss
