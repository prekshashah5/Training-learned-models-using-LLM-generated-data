# LearnedDBComponentsLLM

A unified framework for **learned cardinality estimation** with LLM-based query generation and **active learning** strategies.

This project merges two complementary systems:

- **MSCN-based Cardinality Estimation** with Active Learning (Random, Uncertainty, MC Dropout)
- **LLM-based SQL Query Generation** using LangGraph + Ollama

---

## Pipeline Overview

The diagram below shows the full end-to-end pipeline from query generation through active learning to final evaluation:

![Pipeline Diagram](Thesis/graphs/pipeline.png)

**Stage 1 — Query Generation:** SQL queries are generated via LLM (Ollama) or synthetic procedural generation, then structurally validated against the IMDB schema and converted to MSCN-compatible format.

**Stage 2 — Feature Processing:** Materialized bitmap samples are created from the database. Queries are encoded into feature tensors (one-hot tables/columns/operators/joins + bitmaps) and split into a training pool (90%) and validation set (10%).

**Stage 3 — Active Learning Loop:** The MSCN model is trained iteratively. Each round, an acquisition strategy (Random, Uncertainty, or MC Dropout) selects the most informative unlabeled queries, labels them via PostgreSQL `COUNT(*)`, generates bitmaps, and adds them to the training set.

**Stage 4 — Evaluation:** Q-error metrics are computed, the trained model is compared against the native PostgreSQL optimizer, and a comprehensive dashboard of 15+ graphs is generated.

---

## Project Structure

```
LearnedDBComponentsLLM/
├── config/                         # Unified DB config & settings
│   ├── db_config.py                # DB connection, count_rows, load_column_stats
│   └── settings.py                 # Shared constants & env-var defaults
├── schema/                         # Database DDL & schema files
│   ├── IMDB_schema.txt             # IMDB DDL used for query generation
│   ├── setup_imdb.py               # IMDB data loader
│   └── ...                         # Additional schemas (HR, Movies)
├── mscn/                           # MSCN model architecture
│   ├── model.py                    # SetConv neural network (3 MLPs + output)
│   └── util.py                     # One-hot encoding, normalization utilities
├── generation/                     # Query generation
│   ├── query_generator.py          # LLM & synthetic query generation + SchemaValidator
│   ├── format_converter.py         # SQL ↔ MSCN dict conversion
│   ├── plot_generated_queries.py   # Structural graphs from generated queries
│   └── langraph_ollama/            # LangGraph pipeline (generate → fix → calculate → metrics)
│       ├── main.py                 # StateGraph build & entry point
│       ├── nodes.py                # Node functions (init, generate, fix, cleanup, metrics)
│       ├── state.py                # PipelineState TypedDict
│       ├── generate_queries.py     # Ollama query generation logic
│       ├── fix_queries.py          # LLM-based SQL repair
│       ├── calculate.py            # DB execution of generated queries
│       └── prompt.py               # Prompt templates
├── labeling/                       # Database labeling & bitmap generation
│   ├── db_labeler.py               # Execute COUNT(*) with timeout protection
│   └── bitmap_utils.py             # Materialized sample bitmaps per table
├── training/                       # Model training & active learning
│   ├── pipeline.py                 # Complete end-to-end pipeline (8 steps)
│   ├── train.py                    # MSCN training with AL strategies
│   └── experiment.py               # Quick experiment runner
├── evaluation/                     # Benchmarking & strategy comparison
│   ├── compare_generated_strategies.py  # Compare AL strategies on shared generated queries
│   ├── compare_strategies.py       # Head-to-head strategy comparison (subprocess-based)
│   ├── pipeline_graphs.py          # 15-graph pipeline visualization
│   └── run_benchmarks.py           # Automated benchmarking
├── metrics/                        # LLM query quality metrics
│   ├── main.py                     # Metrics orchestrator
│   ├── compare_models.py           # Cross-model comparison
│   ├── SQL_Complexity.py           # Query complexity scoring
│   ├── selective_non_selective.py   # Selectivity analysis
│   ├── kl_divergence.py            # KL divergence computation
│   ├── validating.py               # SQL validity checks
│   ├── plotting.py                 # Visualization functions
│   └── ...
├── utils/                          # Shared utilities
│   ├── io_utils.py                 # JSON/JSONL/Excel I/O
│   ├── sql_utils.py                # SQL normalization & parsing
│   ├── session_utils.py            # Run directory management
│   └── logger.py                   # Centralized logging
├── tools/                          # Standalone utilities
│   ├── merge_sessions.py           # Merge previous sessions
│   └── get_stats/                  # Column statistics collector
├── prompts/                        # LLM prompt templates
├── docs/                           # Project documentation
├── Thesis/                         # Thesis LaTeX source & result graphs
│   └── graphs/                     # Pipeline diagram + all result PNGs
├── generate_and_plot.py            # One-command query generation + structural plots
├── compare_pg_vs_mscn.py           # Standalone PG vs MSCN comparison
├── .env.example                    # Environment variable template
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Quick Start

### 1. Virtual Environment Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your database credentials and Ollama URL
```

---

## Running the Pipeline

### Training Pipeline (Synthetic Queries — No Ollama Needed)

```bash
python -m training.pipeline --synthetic --total-queries 5000 --rounds 5 --out pipeline_results/synthetic_run
```

### Training Pipeline (LLM-Generated Queries)

```bash
python -m training.pipeline --rounds 5 --out pipeline_results/llm_run
```

By default, this command now reuses the latest file from `generated_queries/` (for example `generated_queries/queries_*.json`) and skips fresh LLM generation when reusable queries exist. If no generated query file is found, it keeps the original behavior and generates new queries.

Useful overrides:

```bash
# Force fresh LLM generation (ignore generated_queries/)
python -m training.pipeline --total-queries 5000 --rounds 5 --no-use-latest-generated --out pipeline_results/llm_run

# Reuse a specific generated query file
python -m training.pipeline --generated-queries-file generated_queries/queries_2026-05-04_10-30-00.json --rounds 5 --out pipeline_results/llm_run
```

### Compare Active Learning Strategies

```bash
python -m evaluation.compare_generated_strategies --strategies random mc_dropout
```

### Generate Queries & Plot Structural Graphs

```bash
python generate_and_plot.py 5000
```

### Plot KL Convergence (Generated vs Reference Workload)

Convert the JOB-light benchmark to JSON first (one-time step):

```bash
python -c "
import json; from pathlib import Path
lines = [l.strip() for l in Path('job-light.sql').read_text(encoding='utf-8').splitlines() if l.strip()]
Path('real_workload').mkdir(exist_ok=True)
Path('real_workload/job_light.json').write_text(json.dumps([{'sql': l} for l in lines], indent=2), encoding='utf-8')
print(f'Converted {len(lines)} queries')
"
```

Then run the comparison:

```bash
python tools/kl_convergence_plot.py \
  --reference real_workload/job_light.json \
  --generated generated_queries/queries_YYYY-MM-DD_HH-MM-SS.json \
  --step 250
```

This command writes three files into `generated_queries/`:

- `*_kl_convergence.csv` — checkpoint-wise KL divergence values (tables, joins, predicates, mean)
- `*_kl_convergence.png` — KL divergence over generation checkpoints (lower = closer to real workload)
- `*_distribution_comparison.png` — side-by-side bar charts comparing JOB-light vs LLM-generated distributions for tables, joins, and predicates

---

## Key Features

| Feature                                                       | Module                                       |
| ------------------------------------------------------------- | -------------------------------------------- |
| MSCN Cardinality Estimation (SetConv)                         | `mscn/model.py`                              |
| Active Learning (Random, Uncertainty, MC Dropout)             | `training/pipeline.py`                       |
| LLM Query Generation (Ollama + Schema Validation)             | `generation/query_generator.py`              |
| Synthetic Query Generation                                    | `generation/query_generator.py`              |
| LangGraph Pipeline (Generate → Fix → Calculate → Metrics)     | `generation/langraph_ollama/`                |
| Shared-Data AL Strategy Comparison                            | `evaluation/compare_generated_strategies.py` |
| Automated Benchmarking                                        | `evaluation/run_benchmarks.py`               |
| Query Quality Metrics (Validity, Complexity, Selectivity, KL) | `metrics/`                                   |
| 15-Graph Pipeline Visualization                               | `evaluation/pipeline_graphs.py`              |
| PostgreSQL vs MSCN Estimator Comparison                       | `compare_pg_vs_mscn.py`                      |
| One-Command Query Gen + Plotting                              | `generate_and_plot.py`                       |

---

## Database Setup

This project requires a **PostgreSQL** database with the **IMDB** dataset.

- DDL schema files are in `schema/` (see `IMDB_schema.txt`)
- Use `schema/setup_imdb.py` to load the data
- The database can also be loaded via `schema/imdb_loader.sh`

---

## Configuration

All runtime parameters can be set via environment variables (`.env` file) or CLI arguments. See `.env.example` for the full list.

| Parameter                  | Default                  | Description                   |
| -------------------------- | ------------------------ | ----------------------------- |
| `DB_HOST`                  | `localhost`              | PostgreSQL host               |
| `DB_NAME`                  | `imdb`                   | Database name                 |
| `OLLAMA_URL`               | `http://localhost:11434` | Ollama API endpoint           |
| `TOTAL_QUERIES`            | `5000`                   | Number of queries to generate |
| `AL_ROUNDS`                | `5`                      | Active learning rounds        |
| `AL_ACQUIRE`               | `200`                    | Queries acquired per round    |
| `AL_EPOCHS`                | `10`                     | Training epochs per round     |
| `NUM_MATERIALIZED_SAMPLES` | `1000`                   | Bitmap sample count per table |
| `HIDDEN_UNITS`             | `256`                    | MSCN hidden layer size        |

---

## Output Artifacts

Each pipeline run creates a timestamped directory (e.g., `pipeline_results/synthetic_run/2026-05-04_10-30-00/`) containing:

- `graphs/` — All 15+ PNG visualizations (learning curves, Q-error CDFs, PG vs MSCN comparisons, etc.)
- `model.pt` — Final trained PyTorch model state
- `learning_data.csv` — Per-round labeled size and median Q-error
- `labeling_times.csv` — Per-round labeling time measurements
- `pipeline_config.txt` — Full configuration dump
- `*.bitmap` — Serialized materialized bitmap data
