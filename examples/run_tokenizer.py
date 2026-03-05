# import os
# import sys

# # Add project root to Python path
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(ROOT_DIR)


# from dataset.gpt_dataset import GPTDatasetV1
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# data_path = os.path.join(BASE_DIR, "data", "the-verdict.txt")

# with open(data_path, "r", encoding="utf-8") as f:
#     raw_text = f.read()

# print("Loaded text length:", len(raw_text))
# print(raw_text[:200])
import os
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from tokenizer.bpe_trainer import train_bpe
from dataset.gpt_dataset import GPTDatasetV1
from tokenizer.bpe_tokenizer import BPETokenizer
from dataset.gpt_dataset import create_dataloader_v1

data_path = os.path.join(BASE_DIR, "data", "the-verdict.txt")

with open(data_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Loaded text length:", len(raw_text))

# Train BPE merges
merges = train_bpe(raw_text, num_merges=50)

print("\nFirst 10 learned merges:")
print(merges[:10])

# Initialize tokenizer
tokenizer = BPETokenizer(merges)
sample_text = "lowest newer wider"

# Encode to tokens
tokens = tokenizer.encode(sample_text)
print("\nTokens:")
print(tokens)

# Build vocabulary from tokens
tokenizer.build_vocab(tokens)
print("\nVocabulary:")
print(tokenizer.token_to_id)

# Encode to IDs
ids = tokenizer.encode_to_ids(sample_text)
print("\nToken IDs:")
print(ids)

# Decode from IDs
decoded = tokenizer.decode_from_ids(ids)
print("\nDecoded text:")
print(decoded)

import torch
print("\n--- Creating DataLoader ---")

# Build vocabulary using the entire corpus
corpus_tokens = tokenizer.encode(raw_text)
tokenizer.build_vocab(corpus_tokens)

# Create dataloader
dataloader = create_dataloader_v1(
    raw_text,
    tokenizer,
    batch_size=4,
    max_length=16,
    stride=16
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("\nInput IDs:")
print(inputs)
print("\nTargets:")
print(targets)
print("\nInput shape:", inputs.shape)

# Token embedding layer
vocab_size = len(tokenizer.token_to_id)
embedding_dim = 64
token_embedding = torch.nn.Embedding(vocab_size, embedding_dim)
embedded = token_embedding(inputs)
print("\nEmbedded shape:", embedded.shape)