# spendcube

A config-driven pipeline that turns raw purchase data into a spend cube: for every input record it decides
whether it is real spend (and why not, if not), assigns a supplier key and an item key, computes net spend in
the reporting currency, and classifies it into the company's own category tree with a calibrated confidence.

Built for the Cantill.ai Data & ML task. Two companies were provided (A: mining, two overlapping systems;
B: metal packaging, eight ERPs and many currencies). A third, unseen company (C) is handled by writing a new
config, not new code.

> No data files are in this repository. Put the data in a folder of your own and point the tool at it.


### The one command

```
uv run python -m spendcube run \
    --config configs/company_A.yaml \
    --data-dir path/to/the/data \
    --out outputs/yourname__A.csv
```

This reads the raw files named in the config, runs every stage, writes the output file in the required format,
and validates it (exit code 1 if validation fails). Typical run times: A (65,549 records) about 80-110 seconds,
B (41,028 records) under 15 seconds, both on a laptop, without the LLM.

Useful options:

| Option | Meaning |
|---|---|
| `--labels-dir DIR` | Where the labels file, tree file and FX file live if they are not in `--data-dir` (e.g. your original kit). Files are found by name even if the folder layout differs. |
| `--no-labels` | Ignore labels (zero-label mode). Labels are used for supplier/item lookup and for training the classifier when present. |
| `--llm` | Label items with Groq (see below). Needs `GROQ_API_KEY`. |
| `--llm-model`, `--tpm`, `--llm-limit N` | Model name, tokens-per-minute budget, and a cap on the number of items (cheap test). |

## Output

One row per input record, columns:
`record_id, include, exclusion_reason, supplier_key, item_key, spend, l1, l2, l3, l4, confidence`

* `exclusion_reason` is one of `CANCELLED`, `NOT_APPROVED`, `NON_PROCUREMENT`, `DUPLICATE` (blank when included).
* `spend` is net, in the reporting currency, and exactly 0 for excluded rows.
* `l1..l4` are the tree's own names, blank below the lowest level and blank for excluded rows.
* `confidence` is 0-1. For included rows it is the (calibrated) classification confidence; for excluded rows it is
  how sure the exclusion decision is (e.g. 0.99 for a status, 0.98 for a PO + amount match, 0.70 for a PO-window match).

`uv run python -m spendcube validate --config ... --data-dir ... --file outputs/x.csv` checks any output file:
columns, one row per input record, valid codes, spend 0 when excluded, and that every category path really exists in the tree.

## Adding a new company (no code changes)

1. Look at the raw file and draft a config:
   ```
   uv run python -m spendcube profile --file data/lines.csv --tree data/tree.csv --company C --currency USD --out configs/company_C.yaml
   ```
   It prints each column (blank %, distinct values, example) and writes a first-guess config with `TODO` comments.
2. Edit the TODOs: id column, amount basis (net/gross), status values, FX table if there are several currencies,
   extra columns that hint at the category (`classify.context_fields`), a one-line domain hint for the LLM.
3. Run `run`, then `validate`. Add company-specific exclusion rules (`exclusions:`) if the data shows them;
   `uv run python -m spendcube exclude ...` prints how often each rule fires and warns about rules that match nothing.

Config blocks (see `configs/company_A.yaml` and `company_B.yaml`): `sources` (file, id, column mapping, amount basis,
status mapping), `fx`, `exclusions`, `cross_system` (only for companies with two overlapping systems), `labels`,
`tree` (any number of levels up to 4), `entities`, `classify`, `output`. Generic exclusion rules (tax labels,
intercompany, royalties, interest) apply to every company from `spendcube/default_rules.yaml`.

## Classification and the LLM

Three situations, in order of preference:

1. **Labels exist** (A2/B2 with the original labels): a linear model on text + supplier + context fields, with
   confidence calibrated on held-out folds. Optionally the LLM re-labels only the items the model is unsure about.
2. **No labels, LLM available** (company C): the LLM labels each *unique item* against the tree, largest spend first;
   a supplier-context model trained on those labels fills the gaps and acts as a second opinion.
3. **No labels, no LLM** (fully offline fallback): the tree's own words plus self-training. This works but is weak
   (see APPROACH.md); it exists so the tool always produces a valid file.

### Groq

```
# PowerShell
$env:GROQ_API_KEY = "gsk_..."
# bash
export GROQ_API_KEY=gsk_...

uv run python -m spendcube run --config configs/company_C.yaml --data-dir data_c --out outputs/yourname__C.csv --llm --llm-model openai/gpt-oss-120b --tpm 200000
```

* The key is read from the environment only; it is never written to disk or committed.
* Answers are cached in `.llm_cache/` (git-ignored), so a re-run or an interrupted run costs nothing extra.
* Big trees are classified level by level so the prompt never carries the whole tree.
* A daily-quota error stops the LLM stage cleanly; the remaining items use the local method.
* Free-tier limits are small (a few thousand tokens per minute and a daily cap); for a full-size file use a plan with
  higher limits and set `--tpm` accordingly. Cost per 1,000 lines is in APPROACH.md.

Optional local embeddings (`--embeddings --embeddings-backend fastembed`, `uv add fastembed`) are implemented but
**off by default**: they scored worse than the plain method on both companies (APPROACH.md, DECISIONS.md D-23).

## Other commands

| Command | Purpose |
|---|---|
| `ingest`, `clean` | Read the files / clean amounts, tax, FX and dates, with a summary. |
| `exclude` | Exclusions and duplicates, rule hit counts; `--labels seed_labels.csv` compares with labelled exclusions. |
| `check-match` | Validates PO-less matching by hiding known POs (run this first on new data for company A). |
| `keys` | Supplier/item keys scored against the golden keys. |
| `classify` | Classification scored against labels: `--holdout 0.2` (train on the rest), `--no-labels` (zero-label), `--llm`. |
| `profile` | Column profile and a draft config for a new company. |

## Layout

```
configs/                 company_A.yaml, company_B.yaml (one file per company)
spendcube/
  config.py ingest.py    config loading, reading files into one canonical record
  clean.py               amounts, VAT, FX, dates
  exclusions.py          status + rules (+ default_rules.yaml) + duplicates
  match.py               tool-to-ERP matching, repeated exports
  entities.py            supplier and item keys
  classify.py llm.py embed.py   classification, Groq layer, optional embeddings
  output.py pipeline.py  output file + validator, the whole pipeline
  evaluate.py profile.py scoring helpers, draft-config helper
outputs/                 the files for the provided companies
APPROACH.md              the method and measured results (one page)
DECISIONS.md             what the data showed and what I decided, with the risks
```

## Reproducibility

The pipeline is deterministic (models are seeded): two runs on the same input give byte-identical files.
Tests cover the parsers, rules, matching, keys, classification paths, the LLM client (against a fake server),
the output validator and the profile helper.

<!-- ## Known limitations

See DECISIONS.md, section "Risks". In short: the supplier-key lookup bets that the hidden truth is built like the
provided labels; the PO-window matching relies on tool record ids rising with PO number (checked by `check-match`);
zero-label quality without the LLM is low; LLM errors can be systematic. -->

 