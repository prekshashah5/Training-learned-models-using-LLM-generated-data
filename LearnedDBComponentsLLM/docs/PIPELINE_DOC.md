# Active Learning Cardinality Estimation Pipeline

A fully parameterized, schema-agnostic pipeline that automates the complete workflow from **SQL query generation** → **active learning** → **MSCN model training** → **comprehensive analysis**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        pipeline.py (Orchestrator)                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STEP 1: Database Connection                                     │
│      └── db_labeler.py (get_connection)                          │
│                                                                  │
│  STEP 2: Materialized Samples + Auto PK Detection               │
│      └── bitmap_utils.py (get_primary_keys, create_samples)      │
│                                                                  │
│  STEP 3: Query Generation (LLM or Synthetic)                    │
│      └── query_generator.py (Ollama / Synthetic fallback)        │
│          └── validate_sql() filters malformed queries            │
│                                                                  │
│  STEP 4: SQL → MSCN Format Conversion + Vocabulary Building     │
│      └── format_converter.py (parse_sql_to_mscn)                │
│                                                                  │
│  STEP 5: Label Validation Set (DB COUNT(*))                     │
│      └── db_labeler.py (label_queries)                           │
│                                                                  │
│  STEP 6: Active Learning Loop                                    │
│      ├── Encode → Train MSCN → Evaluate                         │
│      ├── Acquire (random / uncertainty / mc_dropout)             │
│      ├── Label acquired queries                                  │
│      └── Generate bitmaps → Repeat                               │
│                                                                  │
│  STEP 7: Save Results (CSV, model, config)                      │
│                                                                  │
│  STEP 8: Comprehensive Graph Generation (14 graphs)             │
│      └── pipeline_graphs.py                                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Main orchestrator - ties all steps together |
| `query_generator.py` | LLM-based (Ollama) and synthetic SQL query generation |
| `format_converter.py` | SQL → MSCN CSV format converter (dynamic alias parsing) |
| `db_labeler.py` | Database labeling with timeout, retry, and error handling |
| `bitmap_utils.py` | Runtime bitmap generation with auto PK detection |
| `pipeline_graphs.py` | Comprehensive graph generation (14 plots) |
| `mscn/model.py` | MSCN (Multi-Set Convolutional Network) model |
| `mscn/util.py` | Encoding utilities (one-hot, set encoding, normalization) |

---

## Quick Start

### Prerequisites

- **Python 3.8+**
- **PostgreSQL** with your target database loaded
- **Ollama** running (only needed for LLM mode, not synthetic)

### Install Dependencies

```bash
pip install torch numpy matplotlib psycopg2-binary requests sqlglot
```

### Run with Synthetic Queries (No Ollama Needed)

```bash
python pipeline.py --synthetic --total-queries 500 --rounds 5 --acquire 50 --epochs 10 --strategy mc_dropout --seed 42
```

### Run with LLM Generation (Ollama)

```bash
python pipeline.py \
  --total-queries 2000 \
  --schema-file ../TryingModels/schema/IMDB_schema.txt \
  --model-name llama3.2 \
  --ollama-url http://localhost:11434 \
  --rounds 10 \
  --acquire 200 \
  --epochs 50 \
  --strategy mc_dropout \
  --seed 42
```

### Use a Custom Database

```bash
python pipeline.py \
  --schema-file /path/to/your_schema.sql \
  --db-name your_database \
  --db-user your_user \
  --db-password your_password \
  --total-queries 1000 \
  --rounds 5 \
  --acquire 100 \
  --epochs 20
```

---

## Pipeline Steps Explained

### STEP 1: Database Connection

Establishes a connection to the PostgreSQL database using `psycopg2`. The connection is configured with `autocommit = False` so that transactions are explicitly managed - each successful query label is committed individually, and failed queries are rolled back cleanly.

**Module**: `db_labeler.get_connection()`  
**Input**: `--db-host`, `--db-port`, `--db-name`, `--db-user`, `--db-password`  
**Output**: A live `psycopg2` connection and cursor used throughout the pipeline

The connection is kept alive for the entire pipeline run and shared across labeling, bitmap generation, and sample creation steps.

---

### STEP 2: Materialized Samples + Auto PK Detection

This step prepares the data needed for runtime bitmap generation by:

1. **Auto-detecting primary keys**: Queries `information_schema.table_constraints` for explicit `PRIMARY KEY` constraints. If none are found (common in imported datasets), it falls back to using the **first column** of each table via `information_schema.columns` with `ordinal_position = 1`.

2. **Creating materialized samples**: For each table, selects `N` random primary key values (default 1000) using `ORDER BY RANDOM() LIMIT N`. These sampled PKs form the basis of bitmap vectors - during bitmap generation, each query's predicates are tested against these sample rows.

**Module**: `bitmap_utils.get_primary_keys()`, `bitmap_utils.create_materialized_samples()`  
**Input**: Database cursor, `--num-materialized-samples`  
**Output**: `table_primary_keys` dict (e.g., `{"title_basics": "tconst", ...}`) and `materialized_samples` dict (table → numpy array of sampled PKs)

**Why materialized samples?** MSCN requires fixed-size bitmap features per query. Instead of pre-computing bitmaps for all possible queries offline, we sample a fixed set of rows once, then generate bitmaps on-the-fly by checking which sampled rows satisfy each query's predicates. This makes the pipeline flexible to any generated workload.

---

### STEP 3: Query Generation

Generates `--total-queries` SQL queries using one of two modes:

**LLM Mode (default):**
- Reads the schema DDL from `--schema-file` and optional column stats from `--stats-file`
- Constructs a dynamic prompt with strict rules: only `SELECT COUNT(*)`, only `AND` conditions, only `=`, `<`, `>` operators, only numeric values, no `OR`/`IS NULL`/`LIKE`/subqueries
- Sends the prompt to Ollama in batches of `--batch-size-gen` queries
- Parses the JSON response to extract individual SQL statements
- Each raw SQL is validated by `validate_sql()` which rejects malformed queries before they ever reach the database

**Synthetic Mode (`--synthetic`):**
- Generates structured query dicts directly in Python without any external service
- Randomly combines tables, joins, and predicates from generic table names
- Useful for testing the pipeline mechanics without needing Ollama or a real schema

**Module**: `query_generator.generate_all_queries()`, `query_generator.generate_synthetic_queries()`, `query_generator.validate_sql()`  
**Input**: `--total-queries`, `--schema-file`, `--stats-file`, `--model-name`, `--ollama-url`  
**Output**: List of raw SQL strings (LLM mode) or structured query dicts (synthetic mode)

---

### STEP 4: SQL → MSCN Format Conversion + Vocabulary Building

Converts raw SQL strings into the structured format that the MSCN model requires. Each query is decomposed into three components:

1. **Tables**: `["title_basics tb", "title_ratings tr"]` - full table name + alias
2. **Joins**: `["tb.tconst=tr.tconst"]` - equi-join conditions
3. **Predicates**: `[("tb.startyear", ">", "2000"), ("tr.averagerating", "<", "8")]` - column, operator, value triples

The parser dynamically extracts table aliases from the `FROM` clause using regex (no hardcoded alias map). It also queries the database for each column's `MIN`/`MAX` values to build scaling metrics for numeric features.

After parsing all queries, the pipeline builds vocabularies:
- **table2vec**: One-hot encoding for each unique table
- **column2vec**: One-hot encoding for each unique column
- **op2vec**: One-hot encoding for operators (`=`, `<`, `>`)
- **join2vec**: One-hot encoding for each unique join pattern

The queries are then split into a **pool** (for active learning) and a **validation set** (for evaluation).

**Module**: `format_converter.parse_sql_to_mscn()`, `format_converter.build_column_min_max_from_db()`  
**Input**: Raw SQL strings, database cursor  
**Output**: List of structured query dicts, vocabulary mappings, column_min_max scaling dict

---

### STEP 5: Label Validation Set

Labels the validation set by executing each query on the database to get the true cardinality:

1. Reconstructs the SQL from structured components: `SELECT COUNT(*) FROM table1 t1, table2 t2 WHERE t1.id = t2.fk AND t1.col > 100`
2. Sets a per-query `statement_timeout` (default 6s) to prevent long-running queries from blocking
3. Executes the query and captures the `COUNT(*)` result
4. Resets `statement_timeout = 0` after each execution to prevent timeout leaks
5. If execution fails (timeout, missing table, syntax error), retries up to 2 times with backoff
6. If all retries fail, defaults the cardinality to `1` (MSCN requires cardinality ≥ 1)

Bitmaps are also generated for the validation set by checking each query's predicates against the materialized sample rows.

**Module**: `db_labeler.label_queries()`, `bitmap_utils.generate_bitmaps_for_queries()`  
**Input**: Validation query dicts, database cursor, `--db-timeout`  
**Output**: Validation queries with `cardinality` field populated, validation bitmaps

---

### STEP 6: Active Learning Loop

The core training loop that iteratively improves the model by strategically selecting which queries to label next:

**Initialization:**
- Selects an initial batch of `--acquire` queries from the pool
- Labels them (executes on DB) and generates their bitmaps
- Initializes the MSCN model with `--hid-units` hidden units and Adam optimizer (lr=0.0001)

**Each Round (repeated `--rounds` times):**

1. **Encode**: Converts all labeled queries into MSCN tensor format using the vocabularies and bitmaps. Each query becomes three feature vectors (samples, predicates, joins) with corresponding masks.

2. **Train**: Trains the MSCN model for `--epochs` epochs using Q-error loss. The Q-error loss is defined as `max(pred/true, true/pred)` - it penalizes both over- and under-estimation symmetrically on a ratio scale.

3. **Evaluate**: Runs the trained model on the validation set and computes Q-error metrics (median, 90th, 95th percentile). These are tracked across rounds to produce the learning curve.

4. **Acquire**: Selects the next batch of queries from the unlabeled pool using the chosen strategy:
   - **Random**: Uniformly samples `--acquire` queries
   - **Uncertainty**: Selects queries whose predictions are closest to 0.5 (most uncertain in normalized space)
   - **MC Dropout**: Runs the model T=25 times with dropout enabled, computes variance across predictions, and selects queries with highest variance (most disagreement = most informative)

5. **Label & Bitmap**: Executes the acquired queries on the database and generates their bitmaps

6. **Repeat**: Adds the newly labeled queries to the training set and starts the next round

The loop terminates when all rounds are completed or the unlabeled pool is exhausted.

**Module**: `pipeline.py` (inline), `mscn/model.py` (SetConv), `mscn/util.py` (encoding)  
**Input**: All queries, vocabularies, bitmaps, `--strategy`, `--rounds`, `--acquire`, `--epochs`  
**Output**: Trained model, per-round metrics (labeled_sizes, median_errors, qerrors, epoch losses)

---

### STEP 7: Save Results

Saves all pipeline outputs to a timestamped directory:

- **`learning_data.csv`**: Per-round metrics with columns `labeled_size`, `median_qerror`, `round`
- **`pipeline_config.txt`**: Complete record of all command-line arguments used
- **`model.pt`**: Trained MSCN model weights (can be loaded later with `torch.load`)

**Output Directory**: `pipeline_results/<YYYY-MM-DD_HH-MM-SS>/`

---

### STEP 8: Comprehensive Graph Generation

Generates 14 graphs organized into 3 categories, saved in a `graphs/` subdirectory:

- **Data Generation Analysis** (6 graphs): Visualize the structure and distribution of generated queries - how many tables, joins, predicates per query, what cardinality ranges are covered, and how many queries passed/failed validation.

- **Model Training & Testing** (6 graphs): Track training progress and model quality - learning curve showing Q-error improvement as more data is labeled, training loss convergence, predicted vs actual cardinality scatter, Q-error CDF, and per-round error distribution.

- **Analysis Summary** (2 graphs): High-level overview - labeling success rate and a 4-panel summary dashboard combining key metrics into one view.

**Module**: `pipeline_graphs.generate_all_graphs()`  
**Input**: All tracked metrics from the AL loop (queries, losses, qerrors, predictions, labeling stats)  
**Output**: 14 PNG files in `results_dir/graphs/`

---

## Command-Line Arguments

### Query Generation

| Argument | Default | Description |
|----------|---------|-------------|
| `--total-queries` | `5000` | Total number of SQL queries to generate |
| `--batch-size-gen` | `20` | Number of queries per LLM batch |
| `--model-name` | `llama3.2` | Ollama model name for query generation |
| `--ollama-url` | `http://206.1.53.104:11434` | Ollama API URL |
| `--schema-file` | `../TryingModels/schema/IMDB_schema.txt` | Path to DDL schema file |
| `--stats-file` | *(empty)* | Optional column statistics file |
| `--synthetic` | `false` | Use synthetic generation (no Ollama) |

### Database

| Argument | Default | Description |
|----------|---------|-------------|
| `--db-host` | `localhost` | PostgreSQL host |
| `--db-port` | `5432` | PostgreSQL port |
| `--db-name` | `imdb` | Database name |
| `--db-user` | `postgres` | Database user |
| `--db-password` | `1111` | Database password |
| `--db-timeout` | `6000` | Per-query timeout in milliseconds |

### Model & Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-materialized-samples` | `1000` | Bitmap sample count per table |
| `--strategy` | `random` | AL strategy: `random`, `uncertainty`, `mc_dropout` |
| `--rounds` | `5` | Number of active learning rounds |
| `--acquire` | `200` | Queries to acquire per round |
| `--epochs` | `10` | Training epochs per round |
| `--batch-size-train` | `1024` | Training batch size |
| `--hid-units` | `256` | MSCN hidden layer size |
| `--cuda` | `false` | Use GPU if available |

### Output

| Argument | Default | Description |
|----------|---------|-------------|
| `--out` | `pipeline_results` | Output directory |
| `--seed` | `None` | Random seed for reproducibility |

---

## Generated Graphs

### Category 1: Data Generation Analysis

| Graph | Description |
|-------|-------------|
| **Tables Distribution** | Histogram of tables per query |
| **Joins Distribution** | Histogram of joins per query |
| **Predicates Distribution** | Histogram of predicates per query |
| **Structural Features** | Combined 3-panel overview (tables + joins + predicates) |
| **Cardinality Distribution** | Log-scale distribution of true cardinalities |
| **Query Validation** | Pie chart: valid vs rejected (validation/parse) |

### Category 2: Model Training & Testing

| Graph | Description |
|-------|-------------|
| **Learning Curve** | Labeled samples vs median Q-error with improvement annotation |
| **Training Loss** | Per-epoch loss across all AL rounds (color-coded) |
| **Predicted vs Actual** | Log-log scatter plot with ideal y=x line |
| **Q-error CDF** | Cumulative distribution with p50/p90/p95 markers |
| **Q-error Boxplot** | Boxplot per AL round showing error spread reduction |
| **Q-error Stats** | Median, 90th, 95th percentile trends per round |

### Category 3: Analysis Summary

| Graph | Description |
|-------|-------------|
| **Labeling Rate** | Pie chart: successfully labeled vs defaulted to 1 |
| **Pipeline Summary** | 4-panel dashboard (curve + loss + structure + labeling) |

---

## Schema-Agnostic Design

The pipeline works with **any PostgreSQL database** - no hardcoded table or column names:

1. **Query Generation**: The `--schema-file` DDL is injected directly into the LLM prompt, so generated queries match your actual schema.

2. **Primary Key Detection**: `bitmap_utils.get_primary_keys()` queries `information_schema` for explicit PKs; falls back to using each table's first column.

3. **Alias Parsing**: `format_converter.py` dynamically extracts table aliases from SQL `FROM` clauses using regex.

4. **Query Validation**: `validate_sql()` rejects malformed queries (containing `OR`, `IS NULL`, subqueries, etc.) before they reach the database.

---

## Active Learning Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `random` | Randomly samples from the unlabeled pool | Baseline comparison |
| `uncertainty` | Selects queries closest to decision boundary (pred ≈ 0.5) | Simple uncertainty sampling |
| `mc_dropout` | Monte Carlo Dropout (T=25 forward passes); selects highest variance | Best performance, higher compute |

---

## Module Details

### `query_generator.py`

- **LLM Mode**: Sends the schema + stats to Ollama with strict prompt rules (no OR, no IS NULL, numeric predicates only)
- **Synthetic Mode**: Generates structured query dicts directly (no database or Ollama needed)
- **Validation**: `validate_sql()` filters out malformed LLM output before database execution

### `format_converter.py`

- Parses `SELECT COUNT(*)` queries into MSCN format: `{tables, joins, predicates}`
- Dynamically extracts aliases from `FROM` clause
- Queries database for column `MIN`/`MAX` values for scaling

### `db_labeler.py`

- Executes `SELECT COUNT(*)` with configurable per-query timeout
- Retries failed queries (max 2 attempts with backoff)
- Resets `statement_timeout = 0` after every query to prevent timeout leaks
- Distinguishes error types: `Timeout`, `Table not found`, `UndefinedColumn`, etc.

### `bitmap_utils.py`

- Creates materialized samples (random PK values) per table at startup
- Generates per-query bit-vectors by checking predicate satisfaction via SQL
- Auto-detects PKs from `information_schema`; falls back to first column

### `pipeline_graphs.py`

- Self-contained plotting module (no external dependencies beyond matplotlib/numpy)
- Generates all 14 graphs in a `graphs/` subdirectory
- All functions accept raw pipeline data directly

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `WARNING: Could not auto-detect primary keys` | Database tables have no explicit PK constraints | Pipeline falls back to first column automatically |
| `[db_labeler] Table not found` | Schema file doesn't match actual DB tables | Update `--schema-file` to match your database |
| `[db_labeler] Timeout after Xms` | Query too slow (large joins) | Increase `--db-timeout` (e.g., `30000` for 30s) |
| `[skip-validation] Rejected malformed SQL` | LLM generated invalid SQL (OR, IS NULL, etc.) | Normal behavior - filtered automatically |
| Ollama connection error | Ollama not running or wrong URL | Start Ollama or update `--ollama-url` |
| All cardinalities = 1 | Database tables not loaded | Load your dataset into PostgreSQL |
