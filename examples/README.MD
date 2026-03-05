# Mini BPE Tokenizer Engine

A simple implementation of a **Byte Pair Encoding (BPE) tokenizer from scratch** in Python.

This project demonstrates how tokenizers used in large language models work internally.

The implementation includes:

- BPE merge learning from text corpus
- Tokenization (encode)
- Detokenization (decode)
- Vocabulary creation
- Token → ID mapping
- GPT-style dataset generation
- DataLoader pipeline
- Token embeddings with PyTorch

---

# Project Structure

```
Mini-Tokenizer-engine

data/
    the-verdict.txt

dataset/
    gpt_dataset.py

tokenizer/
    bpe_trainer.py
    bpe_tokenizer.py

examples/
    run_tokenizer.py

tests/
```

---

# How BPE Works

Example:

```
lowest
```

Initial tokens:

```
l o w e s t </w>
```

Merge operations:

```
(l,o) → lo
(lo,w) → low
(e,s) → es
(es,t) → est
```

Final tokens:

```
low est
```

---

# Example

Encoding:

```
lowest newer wider
```

Tokens:

```
['low', 'est</w>', 'new', 'er</w>', 'wid', 'er</w>']
```

Token IDs:

```
[2,1,3,0,4,0]
```

---

# Pipeline

```
text
↓
train BPE merges
↓
tokenizer
↓
token IDs
↓
GPT dataset
↓
dataloader
↓
embedding layer
```

---

# Running the Project

From the project root:

```
py examples/run_tokenizer.py
```

---

# Requirements

```
torch
numpy
```

Install with:

```
pip install -r requirements.txt
```