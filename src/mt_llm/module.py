import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.m2m_100.modeling_m2m_100 import (
    M2M100Encoder,
    M2M100Model,
)
from trident.core.module import TridentModule
from trident.utils.logging import get_logger
from typing import Optional
from src.modules.functional import pooling
from src.mt_llm.evaluation import fvu
from transformers.models.llama import LlamaModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from peft.peft_model import PeftModelForFeatureExtraction
from functools import partial

log = get_logger(__name__)
torch.set_float32_matmul_precision("medium")


class NLLBEncoder(nn.Module):
    def __init__(
        self,
        nllb: M2M100Model,
        pooling_strategy: str = "mean",
        padding_side: str = "right",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.nllb: M2M100Encoder = (
            nllb.encoder if isinstance(nllb, M2M100Model) else nllb
        )
        self.pooling_strategy = pooling_strategy
        self.pooling_fn = partial(
            getattr(pooling, self.pooling_strategy), padding_side=padding_side
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        nllb_embeds_NLD = self.nllb(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        pooler_output = self.pooling_fn(nllb_embeds_NLD, attention_mask)
        return BaseModelOutputWithPooling(
            last_hidden_state=nllb_embeds_NLD,
            pooler_output=pooler_output,
        )


class NLLBLlamaEncoder(nn.Module):
    def __init__(
        self,
        llama: LlamaModel | PeftModelForFeatureExtraction,
        nllb: M2M100Model,
        pooling_strategy: str = "mean",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.llama = llama
        self.nllb: M2M100Encoder = nllb.encoder
        self.up_proj = nn.Linear(nllb.config.hidden_size, llama.config.hidden_size)
        self.pooling_strategy = pooling_strategy
        self.pooling_fn = getattr(pooling, self.pooling_strategy)

        for p in self.nllb.parameters():
            p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with torch.inference_mode():
            nllb_embeds_MLD = self.nllb(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            nllb_embeds_MLD = self.up_proj(nllb_embeds_MLD)
        outputs = self.llama(
            inputs_embeds=nllb_embeds_MLD,
            attention_mask=attention_mask,
        )
        pooler_output = self.pooling_fn(outputs.last_hidden_state, attention_mask)
        return BaseModelOutputWithPooling(
            last_hidden_state=outputs.last_hidden_state, pooler_output=pooler_output
        )


class DistillationModule(TridentModule):
    def __init__(
        self,
        pooling_strategy: str = "mean",
        *args,
        **kwargs,
    ) -> None:
        # logs all configs to self.hyperparams
        super().__init__(*args, **kwargs)
        self.pooling_strategy = pooling_strategy
        self.pooling_fn = getattr(pooling, self.pooling_strategy)
        assert (
            self.pooling_fn is not None
        ), "`self.pooling_strategy` must be one of mean, eos, cls"

    def forward(self, batch: dict):
        # use constructed model during validation
        with torch.inference_mode():
            nllb_embeds_MLD = self.model.nllb(
                input_ids=batch["nllb_input_ids"],
                attention_mask=batch["nllb_attention_mask"],
            ).last_hidden_state
            nllb_embeds_MLD = self.model.up_proj(nllb_embeds_MLD)
        return self.model.llama(
            inputs_embeds=nllb_embeds_MLD,
            attention_mask=batch.get("nllb_attention_mask"),
            labels=batch.get("labels"),
        )

    def training_step(  # type: ignore
        self, batch: dict[str, torch.Tensor], batch_idx: int = 0
    ) -> torch.Tensor:
        # original model input
        self.model.llama.disable_adapter_layers()
        with torch.inference_mode():
            # through Llama 3 w/o LoRA
            # potentially task fine-tuned
            # base_outputs.last_hidden_state is (N, L, D)
            llama_outputs = self.model.llama(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )

            # self.model.nllb is encoder of NLLB
            nllb_embeds_MKd = self.model.nllb(
                input_ids=batch["nllb_input_ids"],
                attention_mask=batch["nllb_attention_mask"],
            ).last_hidden_state  # (M, L, d)
        # up-projection to llama dimensionality
        # nllb_embeds_MKd ->  nllb_embeds_NKD
        nllb_embeds_MKD = self.model.up_proj(nllb_embeds_MKd)

        self.model.llama.enable_adapter_layers()
        nllb_llama_outputs = self.model.llama(
            inputs_embeds=nllb_embeds_MKD, attention_mask=batch["nllb_attention_mask"]
        )

        nllb_hidden_states = self.pooling_fn(
            nllb_llama_outputs.last_hidden_state,
            attention_mask=batch["nllb_attention_mask"],
        )
        # CLS, EOS, Mean pooling
        llama_hidden_states = self.pooling_fn(
            llama_outputs.last_hidden_state, attention_mask=batch["attention_mask"]
        )
        mse_loss = F.mse_loss(nllb_hidden_states, llama_hidden_states)
        with torch.no_grad():
            fvu_loss = fvu(
                x=nllb_hidden_states, x_hat=llama_hidden_states, mse_loss=mse_loss
            )
        self.log("train/mse", mse_loss)
        self.log("train/fvu", fvu_loss)
        return mse_loss


class SpanDistillationModule(DistillationModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        from transformers import AutoTokenizer

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", padding_side="right"
        )
        self.nllb_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M"
        )

    def training_step(  # type: ignore
        self, batch: dict[str, torch.Tensor], batch_idx: int = 0
    ) -> torch.Tensor:
        # original model input
        self.model.llama.disable_adapter_layers()
        with torch.inference_mode():
            # through Llama 3 w/o LoRA
            # potentially task fine-tuned
            # base_outputs.last_hidden_state is (N, L, D)
            llm_outputs = self.model.llama(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )

            # self.model.nllb is encoder of NLLB
            nllb_embeds_MKd = self.model.nllb(
                input_ids=batch["nllb_input_ids"],
                attention_mask=batch["nllb_attention_mask"],
            ).last_hidden_state  # (M, L, d)
        # up-projection to llama dimensionality
        # nllb_embeds_MKd ->  nllb_embeds_NKD
        nllb_embeds_MKD = self.model.up_proj(nllb_embeds_MKd)

        self.model.llama.enable_adapter_layers()
        nllb_llama_outputs = self.model.llama(
            inputs_embeds=nllb_embeds_MKD, attention_mask=batch["nllb_attention_mask"]
        )
        # sequence-level loss
        nllb_seq_embeds = self.pooling_fn(
            nllb_llama_outputs.last_hidden_state,
            attention_mask=batch["nllb_attention_mask"],
        )
        llm_seq_embeds = self.pooling_fn(
            llm_outputs.last_hidden_state, attention_mask=batch["attention_mask"]
        )
        seq_mse_loss = F.mse_loss(nllb_seq_embeds, llm_seq_embeds)
        self.log("train/seq_mse", seq_mse_loss)
        
        # span-level loss
        llm_N, llm_L = batch["input_ids"].shape
        nllb_N, nllb_L = batch["nllb_input_ids"].shape
        nllb_embeds = F.embedding_bag(
            weight=nllb_llama_outputs.last_hidden_state.view(nllb_N * nllb_L, -1),
            input=batch["nllb_bag_ids"],
            # the first BOS token is essentially ignored
            padding_idx=0,
        )
        llm_embeds = F.embedding_bag(
            weight=llm_outputs.last_hidden_state.view(llm_N * llm_L, -1),
            input=batch["bag_ids"],
            # the first BOS token is essentially ignored
            padding_idx=0,
        )

        span_mse_loss = F.mse_loss(nllb_embeds, llm_embeds)
        self.log("train/span_mse", span_mse_loss)
        with torch.no_grad():
            self.log(
                "train/seq_abs_diff_norm",
                (
                    llm_seq_embeds.norm(p=2, dim=-1).mean()
                    - nllb_seq_embeds.norm(p=2, dim=-1).mean()
                ).abs(),
            )
            self.log(
                "train/seq_cos_sim",
                F.cosine_similarity(llm_seq_embeds, nllb_seq_embeds).mean(),
            )
            self.log(
                "train/seq_fvu",
                fvu(x=nllb_seq_embeds, x_hat=llm_seq_embeds, mse_loss=seq_mse_loss),
            )
            self.log(
                "train/span_abs_diff_norm",
                (
                    llm_embeds.norm(p=2, dim=-1).mean()
                    - nllb_embeds.norm(p=2, dim=-1).mean()
                ).abs(),
            )
            self.log(
                "train/span_cos_sim",
                F.cosine_similarity(llm_embeds, nllb_embeds).mean(),
            )
            self.log(
                "train/span_fvu",
                fvu(x=nllb_embeds, x_hat=llm_embeds, mse_loss=span_mse_loss),
            )
        # for now we set 1:2 loss ratio, TBD
        mse_loss = 0.333 * seq_mse_loss + 0.667 * span_mse_loss
        self.log("train/mse", mse_loss)
        return mse_loss


class BaseAutoModule(TridentModule):
    def __init__(
        self,
        # simplifies checkpointing to align with custom validation
        nllb_ckpt: None | str = None,
        checkpoint_path: Optional[str] = None,
        save_checkpoint_on_validation_dir: Optional[str] = None,
        pooling_strategy: str = "mean",
        padding_side: str = "right",
        needs_prefix: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.checkpoint_path = checkpoint_path
        self._loaded = False
        self.is_nllb = "NLLB" in str(type(self.model))
        if self.is_nllb:
            self.batch_prefix = "nllb_"
            if isinstance(nllb_ckpt, str):
                ckpt = torch.load(nllb_ckpt, map_location="cuda:0")["state_dict"]
                self.load_state_dict(ckpt, strict=True)
                log.info(f"Successfully restored {nllb_ckpt.split('/')[-1]} checkpoint")
            else:
                log.info("No checkpoint restored")
        else:
            self.batch_prefix = ""

        if not needs_prefix:
            self.batch_prefix = ""

        self.save_checkpoint_on_validation_dir = save_checkpoint_on_validation_dir
        if self.save_checkpoint_on_validation_dir is not None:
            self._validation_epoch = 0

        self.pooling_strategy = pooling_strategy
        self.pooling_fn = partial(
            getattr(pooling, self.pooling_strategy), padding_side=padding_side
        )

    def on_validation_end(self) -> None:
        """This makes it a lot easier to store checkpoints after Lightning's `trainer.val_check_interval`.

        Warning: this will be triggered by `trainer.num_sanity_val_steps`.
        """
        super().on_validation_end()
        if self.save_checkpoint_on_validation_dir:
            from pathlib import Path

            path = Path(self.save_checkpoint_on_validation_dir).joinpath(
                f"validation-epoch={self._validation_epoch}.ckpt"
            )
            self.trainer.save_checkpoint(path, weights_only=True)
            self._validation_epoch += 1


class AutoModuleForSequenceClassification(BaseAutoModule):
    def __init__(self, num_labels: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.is_nllb:
            if hasattr(self.model, "llama"):
                self.head = nn.Linear(
                    self.model.llama.config.hidden_size, num_labels, bias=False
                )
            else:
                self.head = nn.Linear(
                    self.model.nllb.config.hidden_size, num_labels, bias=False
                )
        else:
            self.head = nn.Linear(self.model.config.hidden_size, num_labels, bias=False)

    def forward(self, batch, *args, **kwargs):
        outputs = self.model(
            input_ids=batch[f"{self.batch_prefix}input_ids"],
            attention_mask=batch[f"{self.batch_prefix}attention_mask"],
        )
        hidden_states = outputs.last_hidden_state
        sequence_embeds = self.pooling_fn(
            hidden_states, batch[f"{self.batch_prefix}attention_mask"]
        )
        logits = self.head(sequence_embeds)
        loss = F.cross_entropy(logits, batch["labels"])
        return {
            "loss": loss,  # type: ignore
            "logits": logits,
            "sequence_embeds": sequence_embeds,
        }


class AutoModuleForSequenceClassificationDistillation(
    AutoModuleForSequenceClassification
):
    def __init__(self, ckpt: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ckpt_ = {
            "weight": torch.load(ckpt, map_location="cuda:0")["state_dict"][
                "head.weight"
            ]
        }
        assert isinstance(self.head, nn.Module)
        self.head.load_state_dict(ckpt_, strict=True)

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        out = self(batch)
        loss = F.mse_loss(out["sequence_embeds"], batch["sequence_embeds"])
        self.log("train/mse", loss)
        return {"loss": loss}


class AutoModuleForMultipleChoice(BaseAutoModule):
    def __init__(self, num_choices: int = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # head as per XLMRobertaForMultipleChoice
        if self.is_nllb:
            if hasattr(self.model, "llama"):
                self.head = nn.Linear(
                    self.model.llama.config.hidden_size, 1, bias=False
                )
            else:
                self.head = nn.Linear(self.model.nllb.config.hidden_size, 1, bias=False)
        else:
            self.head = nn.Linear(self.model.config.hidden_size, 1, bias=False)
        self.num_choices = num_choices

    def forward(self, batch, *args, **kwargs):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Assuming batch and outputs are defined and contain the necessary tensors
        # outputs.last_hidden_state shape: (N, L, D)
        # batch["mean_mask"] shape: (N, L, C)

        # Extract the hidden states from the model output
        hidden_states = outputs.last_hidden_state

        # Extract the mask from the batch
        mask = batch["mean_mask"]

        # Use Einstein summation to compute the weighted sum of hidden states according to the mask
        # This results in a tensor of shape (N, C, D)
        choice_embeds = torch.einsum("nlc,nld->ncd", mask, hidden_states)

        # Compute the sum of the mask along the sequence length L
        # This results in a tensor of shape (N, C)
        mask_sum = mask.sum(1)

        # Divide the summed embeddings by the mask sum to get the average embeddings
        # Ensure broadcasting by adding a singleton dimension to mask_sum
        choice_embeds = choice_embeds / mask_sum[:, :, None]
        logits = self.head(choice_embeds).view(-1, self.num_choices)
        loss = F.cross_entropy(logits, batch["labels"])
        return {"loss": loss, "logits": logits, "choice_embeds": choice_embeds}


class AutoModuleForMultipleChoiceDistillation(AutoModuleForMultipleChoice):
    def __init__(self, ckpt: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ckpt_ = {
            "weight": torch.load(ckpt, map_location="cuda:0")["state_dict"][
                "head.weight"
            ]
        }
        assert isinstance(self.head, nn.Module)
        self.head.load_state_dict(ckpt_, strict=True)

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        out = self(batch)
        loss = F.mse_loss(out["choice_embeds"], batch["choice_embeds"])
        self.log("train/mse", loss)
        return {"loss": loss}
