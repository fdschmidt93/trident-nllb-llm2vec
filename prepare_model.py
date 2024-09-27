import argparse
from pprint import pprint
from transformers import AutoModel, AutoConfig
from peft.peft_model import PeftModel
import torch
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compose and merge LLM2Vec models with optional quantization and LoRA integration."
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        help="Pretrained base model name or path (default: 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp')",
    )

    parser.add_argument(
        "--peft_model",
        type=str,
        default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        help="PEFT model name or path to merge with the base model (default: same as base_model)",
    )

    parser.add_argument(
        "--additional_peft_model",
        type=str,
        default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        help="Additional PEFT model name or path to further merge, such as SimCSE (default: 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised')",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./data/model/llm2vec-sup",
        help="Directory path where the merged model will be saved (default: './data/model/llm2vec-sup')",
    )

    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16", "int8"],
        help="Data type for model weights (default: 'bfloat16')",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to load the model on. Defaults to 'cuda' if available, else 'cpu'.",
    )

    parser.add_argument(
        "--no_trust_remote_code",
        action="store_false",
        dest="trust_remote_code",
        help="Do not trust remote code when loading models. By default, trust_remote_code is True.",
    )

    return parser.parse_args()


def pretty_print_args(args):
    """
    Pretty prints the argparse Namespace.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Convert Namespace to dictionary
    args_dict = vars(args)

    # Choose a pretty print method
    print("\n===== Configuration =====")
    pprint(args_dict, indent=4)
    print("=========================\n")


def main():
    args = parse_arguments()

    # Pretty print the arguments
    pretty_print_args(args)

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Map torch_dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
    }
    torch_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)
    print(f"Using torch dtype: {args.torch_dtype}")

    # Load base model configuration
    print(f"Loading base model configuration from '{args.base_model}'...")
    config = AutoConfig.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )

    # Load base model
    print(f"Loading base model '{args.base_model}'...")
    model = AutoModel.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device,
    )

    # Load and merge first PEFT model
    print(f"Merging PEFT model '{args.peft_model}' into base model...")
    model = PeftModel.from_pretrained(
        model,
        args.peft_model,
    )
    print("Merging and unloading first PEFT model...")
    model = model.merge_and_unload()  # This can take several minutes on CPU

    # Optionally load and merge an additional PEFT model
    if args.additional_peft_model:
        print(
            f"Merging additional PEFT model '{args.additional_peft_model}' into the model..."
        )
        model = PeftModel.from_pretrained(
            model,
            args.additional_peft_model,
        )
        print("Merging and unloading additional PEFT model...")
        model = model.merge_and_unload()  # This can take several minutes on CPU

    # Ensure the save directory exists
    os.makedirs(args.save_path, exist_ok=True)

    # Hack to allow saving the model
    if hasattr(model, "_hf_peft_config_loaded"):
        del model._hf_peft_config_loaded

    # Save the merged model
    print(f"Saving the merged model to '{args.save_path}'...")
    model.save_pretrained(args.save_path)
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
