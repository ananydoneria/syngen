# Syngen (GAN + AE + LLM-Assisted Parser)

Generate synthetic tabular CSV datasets from user prompts across multiple domains, with:

- Controlled prompt parsing (`rules`, `llm`, `hybrid`)
- Domain-aware checkpoint routing
- Full AutoEncoder + latent GAN backend
- CLI and GUI interfaces
- Kaggle-based validation pipeline

## Features

- Prompt grammar: natural language + optional `key=value` overrides
- Conflict detection with fail-fast parse errors
- Timestamped output CSV files per run
- GAN/AE-first generation workflow
- Optional Gemini/OpenAI rewrite layer to generalize prompt understanding
- Benchmark pipeline across multiple Kaggle datasets

## Project layout

- `main.py`: CLI entrypoint
- `gui_app.py`: desktop GUI
- `prompt_parser/`: parser, domain packs, LLM rewrite layer
- `gan_ae_full/`: full train/infer stack for tabular AE + GAN
- `checkpoint_registry.py`: profile/domain -> checkpoint mapping
- `kaggle_eval/`: benchmark runner + reports
- `configs/`: training presets (`fast`, `balanced`, `high_quality`)
- `docs/`: architecture, limitations, push checklist

## Quick start

### 1) Install dependencies

CPU:

```bash
pip install -r requirements-cpu.txt
```

CUDA (example for cu128):

```bash
pip install -r requirements-gpu-cu128.txt
```

Dev tools:

```bash
pip install -r requirements-dev.txt
```

### 2) Configure environment

```bash
cp .env.example .env
```

Set at least one LLM provider key if using `llm` or `hybrid` mode:

- `GEMINI_API_KEY` (default provider)
- or `OPENAI_API_KEY` with `PROMPT_PARSER_LLM_PROVIDER=openai`

### 3) Run CLI

```bash
python main.py --prompt "Generate 1200 finance customers with credit score 650-780 and 30% defaulted" --backend gan_ae
```

### 4) Run GUI

```bash
python gui_app.py
```

GUI includes:

- Parse preview before generation
- GAN+AE checkpoint routing preview in results
- One-click open for last generated CSV

## Checkpoint routing

Edit `checkpoints/registry.json`:

```json
{
  "_default": "checkpoints/full_gan_ae_demo",
  "general": "checkpoints/full_gan_ae_general",
  "healthcare_v1": "checkpoints/full_gan_ae_healthcare",
  "finance_v1": "checkpoints/full_gan_ae_finance",
  "ecommerce_v1": "checkpoints/full_gan_ae_ecommerce"
}
```

Required checkpoint files:

- `metadata.json`
- `preprocessor.pkl`
- `encoder.pt`
- `decoder.pt`
- `generator.pt`

## Training

Single run:

```bash
python -m gan_ae_full.train --csv path/to/data.csv --out checkpoints/full_gan_ae_finance --device cuda
```

Preset-based runs:

```bash
python -m gan_ae_full.train --csv path/to/data.csv --out checkpoints/full_gan_ae_finance --device cuda --preset configs/train_balanced.json
```

Presets:

- `configs/train_fast.json`
- `configs/train_balanced.json`
- `configs/train_high_quality.json`

## Kaggle validation pipeline

Run full benchmark:

```bash
python kaggle_eval/run_kaggle_validation.py --ae-epochs 180 --gan-epochs 320 --batch-size 512
```

Preflight only:

```bash
python kaggle_eval/run_kaggle_validation.py --preflight-only --expect-gpu
```

Reports generated:

- `kaggle_eval/output/kaggle_validation_report.json`
- `kaggle_eval/output/kaggle_validation_report.md`

Recent smoke-run similarity scores (1 epoch sanity run, not final-quality training):

| Case | Similarity |
|---|---:|
| telco_churn | 0.9234 |
| stroke_prediction | 0.8187 |
| adult_income | 0.8929 |
| heart_failure | 0.8674 |
| credit_card_fraud | 0.8463 |
| pima_diabetes | 0.8019 |
| diabetes_general | 0.8019 |
| **average** | **0.8504** |

## Testing and CI

Run tests:

```bash
pytest -q
```

GitHub Actions CI runs:

- `ruff check .`
- `pytest -q`

See `.github/workflows/ci.yml`.

## Security notes

- Never commit `.env` or API keys.
- If keys were ever shared in chats/history, rotate them before publishing.
- Large checkpoints and runtime outputs are git-ignored by default.

## Additional docs

- [Architecture](docs/ARCHITECTURE.md)
- [Known Limitations](docs/KNOWN_LIMITATIONS.md)
- [Push Checklist](docs/PUSH_CHECKLIST.md)
