# Strategy Comparison Chart Guide

These figures compare strategies on the same generated query set, same train/pool split, and same validation set.

## Key Metric
- Q-error = max(pred/actual, actual/pred). Lower is better.
- Median Q-error summarizes typical error.
- p90/p95 Q-error summarize tail risk (bad-case behavior).

## Files
- `comparison_learning_curves.png`: Median Q-error vs labeled samples (log scale).
- `comparison_round_stats.png`: Round-wise median/p90/p95 Q-error (log scale).
- `comparison_qerror_cdf.png`: Final validation CDF of Q-error (log-x).
- `comparison_actual_validation_output.png`: Actual validation cardinalities vs predictions.
- `comparison_predicted_vs_actual_scatter.png`: Predicted vs actual scatter per strategy.
- `strategy_round_metrics.csv`: Per-round metrics table.
- `strategy_summary.csv`: Final/best summary metrics table.
- `validation_predictions.csv`: Per-query predictions for downstream analysis.

## Final Snapshot
- Random Sampling: final median=4.0437, p90=52.9293, p95=52.9293, labeled=5549
- MC Dropout: final median=1.8144, p90=5.6208, p95=19.1735, labeled=5682
