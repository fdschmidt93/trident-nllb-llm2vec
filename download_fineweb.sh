#!/bin/bash

# Save the current working directory
WD=$PWD

# Create the target directory if it doesn't exist
mkdir -p ./data/datasets/fineweb/

# Clone the repository with LFS files skipped initially
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/HuggingFaceFW/fineweb ./data/datasets/fineweb

# Change directory to the cloned repository
cd ./data/datasets/fineweb/

# Pull the specific LFS files
git lfs pull --include "sample/10BT/*.parquet"

# Return to the original working directory
cd $WD
