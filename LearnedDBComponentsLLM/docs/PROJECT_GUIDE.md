# Project Guide: Technical Deep Dive

This guide provides an in-depth explanation of the technical architecture and logic within the `ActiveLearningSample` project.

## 🧠 Model Architecture (MSCN)

The **Multi-Set Convolutional Network (MSCN)** is designed to estimate the cardinality of Join-Selection-Projection (JSP) queries. It treats a query as a set of features:

### 1. Representation
-   **Tables**: Represented as a set of one-hot vectors.
-   **Joins**: Represented as a set of vectors identifying the join edges.
-   **Predicates**: Each predicate (e.g., `info_type.id = 5`) is encoded as:
    -   `[column_id (one-hot), operator_id (one-hot), value (normalized)]`.
-   **Samples (Bitmaps)**: We use materialized samples (bitmaps) to represent the intermediate results of the query.

### 2. Multi-Set Processing
The model uses separate MLPs for tables, predicates, and joins. Since each query can have a varying number of these elements, the model:
1.  Applies the same MLP to each element in the set (e.g., all 5 predicates go through the same MLP).
2.  Applies a **Mask** to ignore padding.
3.  Calculates the **Weighted Average** of the hidden states for each set.
4.  Concatenates the three resulting vectors into a single query embedding.
5.  Passes the embedding through an output MLP with a Sigmoid activation (for normalized log-cardinality).

---

## 🔄 Dataflow

1.  **Ingestion**: `load_data` reads `.csv` query files and `.bitmaps` binary files.
2.  **Normalization**: Cardinality labels are transformed: `y_norm = (log(y) - min_log) / (max_log - min_log)`.
3.  **Encoding**: Queries are converted into fixed-size tensors with padding.
4.  **Loop**:
    -   Training: Optimizes Q-error loss using Adam.
    -   Strategy: Acquisition logic ranks unlabeled pool samples.
    -   Prediction: Inverse transform brings normalized outputs back to cardinality scale.

---

## 🔬 Active Learning Strategies

### Uncertainty (Q-Error)
This strategy relies on the model's own predicted error. It selects samples where the model's current "best guess" on a separate unlabeled validation set indicates it is struggling. 

### MC Dropout (Bayesian Uncertainty)
Monte Carlo Dropout approximates Bayesian inference. By keeping Dropout active during inference (`model.train()`):
-   Each forward pass randomly disables neurons.
-   If the model is confident, different neuron combinations will yield similar results (Low Variance).
-   If the model hasn't seen similar data, different neurons will disagree (High Variance).
-   **High Variance = High Uncertainty**.

---

## 📈 Metric: Symmetric Q-Error

Unlike Mean Squared Error (MSE), which over-penalizes errors in large cardinalities, **Q-Error** is symmetric and scale-invariant. 

$$Q(p, a) = \max\left(\frac{p}{a}, \frac{a}{p}\right)$$

-   A prediction of 10 for an actual 100 has a Q-error of 10.
-   A prediction of 1000 for an actual 100 has a Q-error of 10.
-   **Goal**: Minimize the median and 95th percentile Q-error.

---

---

## 🔒 Reproducibility

To ensure fair benchmark comparisons, we implement strict seeding via the `set_seed` function in `train.py`. This guarantees:
1.  **Deterministic Initialization**: Model weights start at the same random state.
2.  **Consistent Data Splits**: The random shuffling of query data into train/test/validation sets is identical across runs.
3.  **Repeatable Sampling**: Strategies like `random` and `mc_dropout` will make the same sequence of choices given the same inputs.

---

## 🧪 Running Benchmarks

`run_benchmarks.py` uses `itertools.product` to generate a matrix of experiments:
1.  It iterates through combinations of `Queries`, `Epochs`, and `Rounds`.
2.  For each combination, it calls `compare_strategies.py`.
3.  `compare_strategies.py` runs `random`, `uncertainty`, `mc_dropout`, and `supervised`.
4.  Results are aggregated into a single tabular CSV for easy comparison of data efficiency across different hyperparameter regimes.
