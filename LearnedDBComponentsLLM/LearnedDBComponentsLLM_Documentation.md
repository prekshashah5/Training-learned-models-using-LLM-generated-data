# LearnedDBComponentsLLM
**Master Project Documentation - Extended & Detailed Depth**

---

## 1. Core Architectural Paradigm

**LearnedDBComponentsLLM** replaces standard PostgreSQL Optimizer equations with Deep Learning inferences. Standard optimizers utilize 1D histograms and make independence assumptions about queried data (i.e. assuming $P_{filter}(A \text{ and } B) = P(A) \times P(B)$), which catastrophically underestimate correlated multi-column joins causing poor query execution paths. 

This repository leverages the **Multi-Set Convolutional Network (MSCN)**. To train the MSCN efficiently without spending weeks collecting billions of database query benchmarks, an **Active Learning Loop** selects only the most highly uncertain SQL topologies to train the model on, thereby achieving peak estimation accuracy with a minimum boundary of actual SQL `EXPLAIN / COUNT(*)` ground-truth testing.

---

## 2. Directory Mechanics & Detailed Data Flow

### A. Data Generation (`generation/`)
*   `query_generator.py`: Generates unbounded SQL shapes targeting the IMDB schema.
    *   **Synthetic Logic**: Procedurally constructs valid foreign-key networks. It scans `IMDB_schema.txt` to find matching tuples like `title_basics.tconst = title_crew.tconst` and deterministically stacks SQL strings, appending randomly randomized float/text restrictions on columns (e.g. `runtime_minutes < 152`).
    *   **LLM Pipeline** (`call_ollama`): Interfaces via REST with local LLM instances (Llama 3.2). Through `GENERATION_PROMPT`, it pushes the physical schema into the context-window and injects dynamically rotating `diversity_hints` ("Focus on 3-table join queries", "Focus on single point lookups") to coerce the AI to output JSON arrays of SQL texts simulating real-world workloads.
*   `format_converter.py`: Transforms raw `SELECT COUNT(*) FROM X, Y WHERE ...` into a nested dictionary map:
    ```python
    {
      "tables": ["title_basics tb", "title_crew tc"], 
      "joins": ["tb.tconst = tc.tconst"], 
      "predicates": [("tb.runtime_minutes", "<", "150")]
    }
    ```

### B. Database Mapping (`labeling/`)
*   `db_labeler.py`: The execution barrier. Wraps database hooks inside a multi-threaded timeout loop. Queries that cause `statement_timeout = 60000ms` limits to blow up (indicicative of cartesian cross-joins) are caught and defaulted to `1` cardinality to prevent pipeline failure. Extensively utilized to generate the true labels `y`. Also extracts raw JSON dicts from PostgreSQL `EXPLAIN` natively to map native baseline metrics (`get_pg_estimates`).
*   **`bitmap_utils.py` (The Materialized Knowledge Engine)**: 
    *   Deep Learning struggles with naked string logic. `bitmap_utils.py` remedies this by pulling a static subset from the live DB (e.g., $N=1000$ hardcoded rows per table `create_materialized_samples()`). 
    *   When evaluating a generated query, the script queries PG dynamically targeting specific PKS: `SELECT pk FROM table WHERE {predicates} AND pk = ANY(%s)`. 
    *   It then builds a boolean mask vector (the "Bitmap") sizing $N$. If row $i$ survived the SQL filters natively, the bitmap index $i=1.0$, otherwise $0.0$. This allows the model to "see" exactly how brutal or permissive a filter expression is numerically.

### C. The Convolution Set Layer (`mscn/`)
*   `model.py`: Implements `class SetConv(nn.Module)`. Unlike feed-forward networks, queries are not statically sized (some have 1 predicate, some have 20). 
    *   The model accepts three distinct Multi-Layer Perceptrons (MLPs).
    *   **Averages (Set Aggregation)**: The output features for each sample, predicate, and join are multiplied by a `mask tensor` (to ignore $0$-padding) and mathematically averaged down dimension $1$ (`torch.sum(hid) / mask.sum()`). This guarantees the model flattens sets into fixed-sized representations independent of how many joins the SQL possessed. 
    *   The combined outputs are concatenated into a central fully-connected hidden module with `Dropout` yielding a Sigmoid normalized output.
*   `util.py`: Implements aggressive one-hot encodings mapping every single unique `Operator`, `Column`, and `TableName` generated in the pool into fixed vocabulary sizes matrices for tensor injection.

### D. The Active Learning Loop (`training/pipeline.py`)
This runs the primary orchestration script combining all the steps:
1.  **Preparation**: Vocabularies (`column2vec`, `op2vec`) are locked via parsing all generated outputs.
2.  **Dataset Construction** (`make_dataset()`): The raw string sets are padded to `max_joins` lengths. Tensors of dimensionality `[batch_size, max_dims, feature_len]` are materialized for PyTorch. 
3.  **Bootstrapping**: Determines the base starting size (Defaults mathematically to exact `20%` of pool sizes) and runs a purely supervised setup (`epochs=10` natively) to align the network gradients into a structurally stable regime.
4.  **Acquisition Function Application**: The AI identifies what to learn next.
    *   *MC-Dropout Logic*: Passes the multi-thousand Unlabeled Dataset blindly through the PyTorch model 10 times consecutively leaving `Dropout` active. Calculates standard deviation/variance across the 10 predictions per query. A query where predictions vary by factors of $1000x$ is highly unstable to the model and appended to the labeling requirement hook for `acquire` extraction.
    *   *Ensemble Logic*: PyTorch instantiates $K=3$ separate networks originating with different randomized normal weights. They train natively. Queries having massive standard deviation cross-model estimations are targeted.
5.  **Round Progression**: Loops for $R$ rounds continually enriching the dataset with the hardest edge cases.

### E. Analytics & Terminal Extraction (`evaluation/`)
*   `pipeline_graphs.py`: Highly sophisticated visual hooks capturing runtime metrics.
    *   **Q-Error Calculus**: Uses standard metric $Q= \max( \frac{\text{Est}}{\text{True}}, \frac{\text{True}}{\text{Est}} )$. A $Q=1.0$ is flawlessly optimal mathematics. 
    *   **Labeling Efficiency Maps**: Automatically overlays supervised baseline trajectories against the active-learning loss curve. If the `mc_dropout` hits $95\%$ efficiency using $30\%$ of the DB timeframe, it's visibly apparent via the `time_breakdown` visual output arrays.
*   `compare_pg_vs_mscn.py`: Skips the loop and runs a pure `[Postgres Native] vs [Trained Model]` inference test yielding 2 definitive maps explicitly showcasing performance degradation thresholds.

---

## 4. Normalization and Tensor Mechanics Deep Dive

1.  **Cardinality Transformation**: Because row counts fluctuate on exponential curves ($0 \rightarrow 1,000,000,000$), the target variables `$y$` natively passed to Mean Squared Error Loss operations must be logarithmic: $y\_norm = \log_{e}(y + 1)$. This handles potential zeroes without failing out.
2.  **Numerical Constraint Limits**: The logarithmic bounds are further smashed natively via minimum-maximum extraction limits mapping into bounded regions mapping $0.0 \rightarrow 1.0$. The MLP outputs mathematically through a `Sigmoid()` tail matching this normalized boundary.
3.  **Unnormalization**: Output logic `final_preds_unnorm = unnormalize_labels()` reverses operations out using the tracked minimal and maximal bounds captured during Step 1 ensuring metric evaluations remain standard integer dimensions matching user contexts.

---

## 5. Development Customization APIs

To add a completely new generation strategy for the Active AI (say, *Core-Set Diversity Acquisition*):
1. Navigate directly to `training/pipeline.py` inside the `if args.strategy == ...` block at line $680$.
2. Load PyTorch `model.eval()`. Extract the final pooled latent tensors right before the Dense layer in `model.py` (`return out, hid`).
3. Compute a distance metric across the `unlabeled_pool_idx` using native scikit-learn boundaries (like $K-Center$ coverage bounds).
4. Fetch the lowest/highest clustered outputs and stack the integer results tracking exact indexes back to the `acquire` length limit arrays.
