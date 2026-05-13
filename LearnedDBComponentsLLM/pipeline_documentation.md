# Learned Cardinalities Active Learning Pipeline
**Comprehensive Technical Documentation**

The `training.pipeline` application serves as the core orchestration engine for evaluating Learned Cardinality Estimators (specifically the Multi-Set Convolutional Network, or MSCN) under an Active Learning framework. This pipeline completely automates synthetic data generation, ground-truth label aggregation from PostgreSQL, feature tensor encoding, neural network training loops, active query acquisition, and comprehensive graph-based statistical evaluation.

---

## 1. High-Level Architecture & Workflow

The pipeline execution is divided into **8 sequentially strict steps**, architected to mirror an automated ML infrastructure from bare strings to analytical graphs.

### Step 1: Database Connection & State Initialization
*   **Module**: `config.db_config`
*   The pipeline establishes a persistent connection to a live PostgreSQL instance (default schema: `imdb`). Active transactions are instantiated to perform rapid operations (like `COUNT(*)`). Global seeds and Torch devices are fixed here.

### Step 2: Materialized Bitmap Calculation
*   **Module**: `labeling.bitmap_utils`
*   **The Problem:** Deep Learning models struggle with naked strings like `"runtime_minutes > 150"`. 
*   **The Solution:** The pipeline pre-samples a random static subset (e.g., 1000 rows) of each baseline table connected by primary keys.
*   When evaluating a generated query, the script queries PG dynamically targeting these specific PKs: `SELECT pk FROM table WHERE {predicates} AND pk = ANY(%s)`. 
*   It then builds a boolean mask vector (the "Bitmap"). If row `i` survived the SQL filters natively, the bitmap index $i=1.0$, otherwise $0.0$. This allows the model to "see" mathematically exactly how permissive a filter expression is.

### Step 3: Neural Query Generation (Unlabeled Pool)
*   **Module**: `generation.query_generator`, `generation.format_converter`
*   To train the AI, thousands of syntactically valid relational queries must be generated.
    *   **Synthetic Mode (`--synthetic`)**: Fast, deterministic structural generation that recursively joins basic tables, randomly selects column filters, and guarantees valid foreign key paths based on a hardcoded schema constraint mapping.
    *   **LLM Mode (`--model-name ollama`)**: Generates rich, highly creatively varied queries via prompting an external local Language Model (e.g., Llama 3) via REST. Pushes the physical schema into the context-window and injects dynamically rotating `diversity_hints`.
*   All string SQLs are parsed out into Python dicts tracking `tables`, `joins`, and `predicates`.

### Step 4: Initial Split & Validation Baseline
*   **Module**: `labeling.db_labeler`
*   Queries are cleanly partitioned: $90\%$ **Unlabeled Pool** and $10\%$ **Validation Set**. 
*   The validation subset undergoes physical labeling immediately. PostgreSQL executes `SELECT COUNT(*) FROM ...` loops.
*   **Timeouts & Protections**: A timeout (e.g., `5000ms`, `60000ms`) is enforced natively inside PostgreSQL (`SET statement_timeout`) to dynamically abort massive cartesian cross-joins without halting the Python pipeline globally.

### Step 5: Encoding, Normalization, & Masking
*   **Module**: `mscn.util`
*   Vocabulary lists are programmatically constructed tracking every unique structural permutation.
*   **Feature Tensors**: Categorical data is translated to `one-hot` tensors.
*   **Log-scaling**: True integer cardinalities vary on exponential curves ($0 \rightarrow 1,000,000,000$). The labels $y$ are transformed to $\log_{e}(y + 1)$, and further crushed via Min-Max boundaries to scale from $0.0 \rightarrow 1.0$ so the `MSELoss` gradients do not violently explode.

### Step 6: The Active Learning Loop
This is the iterative focal point of the system. Rather than receiving $N$ labeled queries all at once (supervised baseline), the active learning framework begins with only a tiny statistical fraction and *computationally reasons* about which queries it should ask the labeler (Database) to solve next.

1.  **Bootstrapping Initialize**: The system takes exactly $20\%$ of `--total-queries` randomly and uses them to stabilize the initial model weights natively.
2.  **Epoch Alignment**: The MSCN architecture evaluates backpropagation mapping string/bitmap tensors against Cardinalities (using flattened `nn.Linear` outputs passed via sigmoids).
3.  **Active Acquisition Evaluation**: The model evaluates the Unlabeled sequence to figure out what it *doesn't mathematically comprehend*.
    *   **Random**: Default baseline stochastic sampling.
    *   **MC-Dropout (Monte Carlo Variance)**: A Bayesian Uncertainty computation. Normally, Dropout randomly destroys weights only during training. In MC-Dropout inference, the module passes the Unlabeled Pool through the network $10$ times iteratively with Dropout active. Queries exhibiting massive target variations (high $\sigma^2$) are mathematically determined to be inherently 'Uncertain' to the model state and acquired for specific labeling.
    *   **Ensembles**: Creates a battery of $K$ separate neural models. The lowest variance agreements between models dictate query difficulty edge-cases.
4.  **Append & Restructure**: Acquired queries get labeled natively, bitmapped, and merged dynamically into the main PyTorch `DataLoader`.

### Step 7: Final Graphing & Distributive Logging
*   **Module**: `evaluation.pipeline_graphs`
*   The pipeline evaluates final Q-Errors using standard metrics $Q= \max( \frac{\text{Est}}{\text{True}}, \frac{\text{True}}{\text{Est}} )$.
*   Every active iteration creates $15+$ visually aesthetic PDF/PNG mathematical outputs targeting:
    *   **Labeling Efficiency Maps**: Automatically traces Supervised baselines against the active-learning loss curve mapping exact temporal bounds ($10\%$ pool yielding $95\%$ performance).
    *   **Q-Error CDF Sweeps**: Cumulative Distributive Function graphs displaying model accuracy dropoffs logarithmicly across test arrays.
    *   **Data Layout Distributions**.

### Step 8: Final Postgres Estimator Comparison
*   The pipeline conducts a unified sequence of standard `EXPLAIN` calls to pull native PostgreSQL statistical estimates across the evaluated validation subset, building a perfect baseline model comparison logic between physical engine implementations and AI inferences. Outputs mapped as `compare_pg_vs_mscn_cdf.png`.

---

## 2. Artifact Output Persistence & Telemetry

A single run creates a globally stamped telemetry directory (`pipeline_results/YYYY-MM-DD_HH-MM-SS/`). Inside, it severely archives:

*   `/graphs/` – Contains all compiled MATPLOTLIB analytics charts natively formatted.
*   `model.pt` – The final PyTorch model state dict from the concluding round. Useful for hot-reloading estimator endpoints.
*   `pipeline_config.txt` – Exhaustive trace dict configuration tracking exact seeds and dimensions.
*   `learning_data.csv` – Serialized trace arrays of model median validation capabilities over time.
*   `labeling_times.csv` – Exact timestamp deltas per learning iteration measuring cost/efficiency tradeoffs.
*   `val_bitmaps.bitmap` & `initial_bitmaps.bitmap` – Intermediate serialized raw dictionary payloads mapping exact material schemas evaluated.

---

## 3. Invocation Configuration & CLI Parameters

You can trigger the pipeline directly via terminal execution. All parameters are optionally defined in `.env` or overridden manually.

```bash
python -m training.pipeline --synthetic --total-queries 2500 --rounds 5 --epochs 10 --strategy mc_dropout --acquire 200
```

### Key CLI Flags:
*   **`--total-queries <int>`**: Dictates the mathematical ceiling mapping the un-labeled pool and evaluation validation set. Standard run uses `2500`.
*   **`--synthetic`**: Boolean flag explicitly disabling LLM generation paths in favor of lightning-fast deterministic native algorithmic synthetics matching predefined schema paths. Default is False.
*   **`--strategy <string>`**: Evaluates Active Learning uncertainty matrices (`random`, `mc_dropout`, `ensemble`).
*   **`--acquire <int>`**: The integer limit bounds actively mapping labeling allocations incrementally at the end of each Round.
*   **`--epochs <int>`**: Local training convergence layers (Deep Learning loops) executed over specific labeled tensors per active learning round (typically `10` or `20`).
*   **`--db-timeout <int>`**: Natively mapped `ms` enforced limit at PostgreSQL driver bindings.

---

## 4. API Extensions (Scaling the Platform)

To introduce purely novel active learning heuristics (ex. Core-Set Covering, Fisher Information Matrix calculation), a new mathematical strategy function must be dynamically attached underneath the **Active Learning Loop (Acquisition)** in `training/pipeline.py` (near line $680$). 

New algorithms must explicitly interact with PyTorch tensors evaluated over `unlabeled_pool_idx`, evaluate scoring heuristics (like K-Center clustered distances or Log-Determinants), and securely return integer indexing structures matching the length parameter constraints dictated by `--acquire`.
