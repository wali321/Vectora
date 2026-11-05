# Vectora

Vectora is a personal, educational language-model project that implements
decoder-only Transformer experiments in Python and PyTorch. The project is for
learning how tokenization, embeddings, attention, optimization, and
autoregressive generation fit together; it is not intended to compete with
commercial foundation models.

The current planned iteration, **Vectora v1.3**, pairs a 50,000-token
WordPiece tokenizer with a compact decoder-only Transformer. Its source
notebook contains a planned 50,000-update training loop. A separate archived
resume notebook records a completed run, but its saved output conflicts with
the model definition currently stored beside it. Results for the current
source configuration are therefore not claimed here.

## What is Vectora?

Vectora builds and experiments with language models from their component
operations instead of treating a hosted model as a black box. The notebooks
are the primary implementation artifacts; the v1.3 training and preprocessing
workflows are Jupyter notebooks rather than standalone Python scripts.

## Version history

| Version | Repository evidence |
|---|---|
| v1 | Initial GPT-style Transformer experiment in [`LLM's/Vectora-v1.0/v1.ipynb`](<LLM's/Vectora-v1.0/v1.ipynb>). |
| v1.1 | Character-level Shakespeare experiment in [`LLM's/Vectora-v1.1/v1.1.ipynb`](<LLM's/Vectora-v1.1/v1.1.ipynb>). |
| v1.2 | The checked-in [`V1.2-23M.ipynb`](<LLM's/Vectora-v1.2-23M/V1.2-23M.ipynb>) uses a 333-character vocabulary on an 82,215,804-character corpus. It is not a 50,000-token WordPiece model. |
| v1.3 | 50,000-token WordPiece and decoder-only Transformer source in [`model784m.ipynb`](<LLM's/Vectora-v1.3-784M/model784m.ipynb>), with separate tokenizer and preparation notebooks. The current source plan and an archived continuation run have different configurations. |

The values **5.0094** training loss, **4.9585** validation loss, **50,000**
steps, and **51.2 million** token positions appear in the completed output of
`tests/vectoraresume.ipynb`, which resumes a 50,000-vocabulary experiment from
step 36,250. Its saved output reports an untied 28.7M-parameter model, while
the notebook's current source ties the input and output embeddings. The
checked-in v1.2 notebook is a separate 333-character experiment and reports
different results. We therefore preserve the 5.0094 / 4.9585 values as an
**archived, source-mismatched run**, not as a verified result for either the
current v1.3 source plan or the checked-in v1.2 experiment. See
[training documentation](docs/training.md) for the evidence and caveats.

## Features

- PyTorch implementations of causal self-attention and Transformer blocks.
- A custom-trained WordPiece tokenizer with a 50,000-token vocabulary.
- Gradient accumulation and AdamW optimization in the v1.3 notebook.
- DirectML, CUDA, and CPU device selection in the v1.3 notebook.
- Tokenized binary train and validation files for memory-mapped sampling.
- Temperature and top-k sampling in the model's `generate` method.

## Architecture overview

The current v1.3 source uses token embeddings, rotary position information,
four pre-normalized decoder blocks, and a tied language-model head. The
attention is causal, so a position cannot attend to future tokens.

See [the architecture guide](docs/architecture.md) for a Mermaid diagram and
the parameter-count calculation.

## Model specifications

The following describes the **current v1.3 notebook source and planned
configuration**, not a verified training result.

| Specification | v1.3 |
|---|---:|
| Tokenizer vocabulary | 50,000 |
| Embedding width | 256 |
| Transformer blocks | 4 |
| Attention heads | 4 |
| Head dimension | 64 |
| Context length | 128 tokens |
| Feed-forward hidden width | 682 (SwiGLU) |
| Dropout | 0.1 |
| Position encoding | Rotary (RoPE); no learned position table |
| Tied token embedding / output head | Yes |
| Trainable parameters from current source | **15,945,984** |

The parameter count is calculated from the current `VectoraGPT` source. The
notebook's stored output instead says 28,745,984 parameters, which is
inconsistent with its current tied-embedding source. See
[architecture notes](docs/architecture.md#parameter-count) for the calculation.

## Tokenizer

The v1.3 tokenizer is **WordPiece**, implemented with Hugging Face's
`tokenizers` package and `BertPreTokenizer`. Its `BertNormalizer` lowercases
text, cleans text, handles Chinese characters, and leaves accents unstripped.
The configured special tokens are `[UNK]` and `[PAD]`; no `[CLS]`, `[SEP]`,
or end-of-sequence token is configured.

See [the tokenizer guide](docs/tokenizer.md) for training, loading, and
encoding examples.

## Dataset

The corpus is reported as English web-scraped text intended for language-model
pretraining. Do not infer named sources or complete cleaning/deduplication.

| Measure | Project-reported figure | Local corpus / preprocessing evidence |
|---|---:|---:|
| Lines | Approximately 5,000,000 | **4,935,527** lines counted in `corpus.txt` |
| File size | Approximately 826 MB | **846,655,814 bytes** (846.656 MB decimal / 807.434 MiB) |
| Token count | Approximately 784 million | `prepare.ipynb` reports **189,563,561** tokens; separate `tests` uint16 files encode the same count |
| Primary language | English | Reported project metadata; no language-identification step found |

The reported file size and tokenizer-produced token count do not match the
local corpus and saved preprocessing output exactly. Treat 784 million as an
unreconciled estimate until the tokenizer file, input corpus, and counting
procedure are verified together. Web-scraped text can contain duplicate
content, noise, HTML remnants, and inconsistent formatting.

See [data preparation](docs/data-preparation.md) for the actual preprocessing
logic and its validation-split behavior.

## Training configuration

These are planned/source configuration values. They are not evidence that the
v1.3 run completed.

| Setting | Value |
|---|---:|
| `batch_size` | 1 |
| `accumulation_steps` | 32 |
| `block_size` | 128 |
| `max_iters` | 50,000 |
| `learning_rate` / `min_lr` | `3e-4` / `3e-5` |
| `warmup_iters` | 1,000 |
| `eval_interval` / `eval_iters` | 250 / 30 (the interval is not used by the current loop) |
| `checkpoint_interval` | 1,000 |
| `n_embd` / `n_head` / `n_layer` | 256 / 4 / 4 |
| `dropout` | 0.1 |

With gradient accumulation, the effective batch is 32 sequences or 4,096
token positions per optimizer update. A 50,000-update run would therefore
sample 204.8 million token positions; this is not the number of unique corpus
tokens. See [training](docs/training.md) for the implemented schedule,
validation, checkpoint, and resume details.

## Installation

From the project directory, create an environment and install the notebook
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch numpy tokenizers safetensors jupyter
```

Install the PyTorch build appropriate for your platform and hardware. On
supported Windows systems, DirectML can optionally be installed with
`python -m pip install torch-directml`; the notebook otherwise falls back to
CUDA when available or CPU.

## Inference

The v1.3 notebook defines `VectoraGPT.generate(idx, max_new_tokens,
temperature=0.7, top_k=40)`. There is no separate v1.3 inference script or
checkpoint-loading flow in the source notebook. Its single code cell runs the
entire training loop; after that cell completes, the model and tokenizer
remain available in the notebook session for the example in
[inference documentation](docs/inference.md).

## Training from scratch

The v1.3 workflow is notebook-based. Run the tokenizer training notebook,
then the preprocessing notebook, and verify that both binary files exist
before executing the model training notebook:

```powershell
Set-Location ".\LLM's\Vectora-v1.3-784M"
jupyter notebook berttokenizer.ipynb
# After it saves 784million.json:
jupyter notebook prepare.ipynb
# After train.bin and val.bin are created:
jupyter notebook model784m.ipynb
```

The model notebook's single code cell starts training immediately. It falls
back to random dummy data if `train.bin` or `val.bin` is absent and does not
stop automatically; if its startup output shows that fallback, interrupt the
cell immediately. See [training instructions](docs/training.md) for caveats
and the data-count mismatch.

## Resuming training

Resume training is **not implemented in the current v1.3 notebook**. It saves
periodic model/optimizer checkpoints, but the source does not load a
checkpoint or restore the optimizer and step. A separate resume notebook and
checkpoints exist under `tests/`; their recorded model shape does not match
the current tied-embedding source. See
[training documentation](docs/training.md#resume-training) before
using any checkpoint files.

## Repository structure

```text
.
├── README.md
├── docs/
│   ├── architecture.md
│   ├── data-preparation.md
│   ├── inference.md
│   ├── tokenizer.md
│   └── training.md
├── LLM's/
│   ├── README.md
│   ├── Vectora-v1.0/
│   ├── Vectora-v1.1/
│   ├── Vectora-v1.2-23M/
│   └── Vectora-v1.3-784M/
│       ├── berttokenizer.ipynb
│       ├── corpus.txt
│       ├── model784m.ipynb
│       └── prepare.ipynb
└── tests/
    └── ...
```

`tests/` contains additional experiments and artifacts; files there are not
automatically assumed to be the canonical v1.3 implementation.

## Documentation

- [Architecture](docs/architecture.md)
- [Tokenizer](docs/tokenizer.md)
- [Data preparation](docs/data-preparation.md)
- [Training](docs/training.md)
- [Inference](docs/inference.md)
- [Vectora project overview](docs/lmdocs/content/project/index.mdx)
- [Vectora v1.3 research record](docs/lmdocs/content/models/vectora/v1.3.mdx)

## Current limitations

- The v1.3 notebook's saved output is stale relative to its current source: it
  stops at step 35,486 of 50,000, reports 1,024 tokens per step, and reports
  28,745,984 parameters. The current source implies 4,096 tokens per update
  and 15,945,984 tied-embedding parameters.
- A separate `tests/vectoraresume.ipynb` output reaches step 50,000, but its
  recorded 28,745,984-parameter model conflicts with the tied-embedding
  definition now saved in that notebook. Its metrics are archived run
  evidence, not current-source benchmarks.
- The preprocessing notebook reports 189,563,561 tokens, not approximately
  784 million.
- `eval_interval=250` is declared but not applied; validation is evaluated on
  each training iteration.
- Resume logic, a standalone inference script, and an environment lockfile
  are absent from the version directory. Archived checkpoints are stored under
  `tests/` but are not verified as compatible with the current source.
- No source website list, verified deduplication pass, HTML-removal step, or
  reproducible corpus checksum is documented.

## Future improvements

Reconcile the tokenizer-produced token count, refresh notebook outputs only
after a deliberate training run, add versioned configuration/checkpoint
loading, implement periodic evaluation as configured, add tests and a
dependency lockfile, and document dataset provenance and cleaning where
verified.

## Credits and acknowledgements

Vectora is maintained by Syed Wali Haider. The project uses PyTorch and
Hugging Face Tokenizers. Background reading listed in the earlier project
README includes:

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Radford et al., [Language Models are Unsupervised Multitask Learners](https://d4mucfpksy.cloudfront.net/better-language-models/language-models.pdf).
- Andrej Karpathy, [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY).

Project links: [documentation](https://vectoradocs.vercel.app) ·
[GitHub](https://github.com/wali321/Vectora).
