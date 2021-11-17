import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder

from src.llm2vec.modelling_llama import LlamaEncoderModel


class NLLBLlama(nn.Module):
    def __init__(
        self, nllb: M2M100Encoder, llama: LlamaEncoderModel, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.nllb = nllb
        self.llama = llama
