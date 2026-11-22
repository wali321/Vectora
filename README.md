# Vectora

Vectora is an educational language-model project built with Python and PyTorch. It contains a collection of notebook-based experiments exploring character-level and subword language modeling.

The latest experiment is **Vectora v1.3**, which implements a decoder-only Transformer with a custom-trained 50,000-token WordPiece tokenizer.

> Vectora is a learning project, not a production-ready or commercially competitive language model.

## Experiments

| Version | Description | Data | Implementation |
|---|---|---|---|
| v1.0 | Initial GPT-style Transformer experiment | Wizard of Oz | [Vectora-v1.0/v1.ipynb](Vectora-v1.0/v1.ipynb) |
| v1.1 | Character-level language model | 500,000-character text sample | [Vectora-v1.1/v1.1.ipynb](Vectora-v1.1/v1.1.ipynb) |
| v1.2 | Expanded character-level experiment | 82,215,804-character corpus | [Vectora-v1.2-23M/V1.2-23M.ipynb](Vectora-v1.2-23M/V1.2-23M.ipynb) |
| v1.3 | WordPiece tokenizer and compact decoder-only Transformer | 50,000-vocabulary text corpus | [Vectora-v1.3-784M/](Vectora-v1.3-784M/) |

## Vectora v1.3

The v1.3 source is organized into three notebook workflows:

1. `berttokenizer.ipynb` trains a WordPiece tokenizer.
2. `prepare.ipynb` encodes the corpus into binary training files.
3. `model784m.ipynb` defines and trains the Transformer.

### Model configuration

| Setting | Value |
|---|---:|
| Vocabulary | 50,000 |
| Embedding dimension | 256 |
| Transformer blocks | 4 |
| Attention heads | 4 |
| Context length | 128 tokens |
| Feed-forward hidden dimension | 682 |
| Dropout | 0.1 |
| Position encoding | Rotary positional embedding |
| Activation | SwiGLU |
| Normalization | RMSNorm |
| Embedding/output head | Tied |
| Batch size | 1 |
| Gradient accumulation | 32 |
| Maximum updates | 50,000 |

The effective training batch is 32 sequences, containing 4,096 tokens per optimizer update.

The current `VectoraGPT` source contains approximately **15.95 million trainable parameters**.

### Architecture

The model is composed of:

- Tied token embeddings and language-model head
- RMSNorm before attention and feed-forward blocks
- Four-head causal multi-head attention
- Rotary positional embeddings
- SwiGLU feed-forward networks
- Residual connections
- Causal masking
- AdamW optimization
- Cosine learning-rate decay with linear warmup
- Gradient clipping
- Top-k and temperature-based sampling

## Requirements

- Python 3.9 or newer
- PyTorch
- NumPy
- Hugging Face `tokenizers`
- Safetensors
- Jupyter

Optional:

- CUDA for NVIDIA GPU support
- DirectML for supported Windows GPUs
- CPU execution

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch numpy tokenizers safetensors jupyter
```

For a DirectML environment:

```powershell
python -m pip install torch-directml
```

## Training Vectora v1.3

Change to the v1.3 directory:

```powershell
Set-Location ".\Vectora-v1.3-784M"
jupyter notebook
```

Run the notebooks in this order:

1. `berttokenizer.ipynb`
2. `prepare.ipynb`
3. `model784m.ipynb`

### 1. Train the tokenizer

The tokenizer notebook:

- Uses Hugging Face `tokenizers`
- Trains a 50,000-token WordPiece vocabulary
- Applies lowercasing and BERT-compatible normalization
- Reads `corpus.txt`
- Saves the tokenizer as `784million.json`

### 2. Prepare the data

The preparation notebook:

- Loads `784million.json`
- Reads `corpus.txt`
- Encodes text into `uint16` token IDs
- Writes 90% of each chunk to `train.bin`
- Writes 10% of each chunk to `val.bin`

Both binary files should exist before starting training.

> The model notebook falls back to random dummy data if `train.bin` or `val.bin` is missing. Check the notebook output and interrupt execution if it reports this fallback.

### 3. Train the model

The model notebook:

- Supports DirectML, CUDA, and CPU execution
- Uses gradient accumulation
- Evaluates validation loss
- Saves checkpoints every 1,000 updates
- Exports `model.safetensors`
- Exports `vectora_final.pt`

## Generation

The `VectoraGPT.generate` method supports temperature and top-k sampling:

```python
prompt = tokenizer.encode("Vectora").ids
input_ids = torch.tensor([prompt], dtype=torch.long, device=device)

generated_ids = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
    top_k=40,
)

print(tokenizer.decode(generated_ids[0].tolist()))
```

This interface is available only after executing the model notebook.

## Repository structure

```text
.
├── README.md
├── Vectora-v1.0/
│   ├── v1.ipynb
│   └── wizardofoz.txt
├── Vectora-v1.1/
│   ├── v1.1.ipynb
│   └── train.txt
├── Vectora-v1.2-23M/
│   ├── V1.2-23M.ipynb
│   └── train.txt
└── Vectora-v1.3-784M/
    ├── berttokenizer.ipynb
    ├── model784m.ipynb
    ├── prepare.ipynb
    └── corpus.txt
```

Generated files such as `784million.json`, `train.bin`, and `val.bin` may require local regeneration.

## Current caveats

- Training checkpoints and final weights are experimental.
- The v1.3 notebook does not provide a standalone training script.
- Resume support is not implemented in the current v1.3 source.
- Validation is evaluated on every update rather than only at `eval_interval`.
- The project does not currently include a dataset provenance or licensing record.
- The name `784M` refers to the project's intended corpus size. The generated binary files are the source of truth for the actual token count.
- Results should not be compared across notebook versions without reconciling their tokenization, vocabulary, model shape, and datasets.

## Educational goals

Vectora demonstrates:

- Building a tokenizer with Hugging Face Tokenizers
- Encoding text into memory-mapped binary arrays
- Implementing causal self-attention
- Combining RMSNorm and rotary positional embeddings
- Writing SwiGLU Transformer blocks
- Applying gradient accumulation
- Scheduling learning rates
- Saving and exporting model weights
- Implementing autoregressive text generation

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Author

Syed Wali Haider
