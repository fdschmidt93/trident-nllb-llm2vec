from datasets import load_dataset

dataset = load_dataset("parquet", data_files={"train": "./aym_validation.parquet"}, split="train")


x = list(zip(dataset["premise"], dataset["hypothesis"]))
