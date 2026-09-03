# Execution Logs

Index for the `execution_logs/` folder from the 2026-09-03 15:22 benchmark run. The split path ran the six pairs entirely through the live agent, total wall time 227.3 seconds, with 223.0545 seconds of that in 24 LLM calls.

## Main files

- `benchmark_agent_raw.log` (736 lines, ~24K), full interleaved ReAct trace, including the early single-prompt hallucination and the six successful per-pair runs
- `benchmark_stderr.log` (~48 lines, ~7K), runner stderr with per-pair elapsed times and the final `6/6 combinations succeeded` summary
- `benchmark_stdout.log` (~17 lines), the markdown table as printed to stdout
- `benchmark_full_raw.log` (~801 lines, ~32K), convenience concatenation of stderr plus stdout plus raw trace, if you want it in one file

## Per-pair split traces: six logs, one per dataset/algorithm pair

- `benchmark_split_iris_decision_tree.log`
- `benchmark_split_iris_random_forest.log`
- `benchmark_split_iris_pytorch_mlp.log`
- `benchmark_split_breast_cancer_decision_tree.log`
- `benchmark_split_breast_cancer_random_forest.log`
- `benchmark_split_breast_cancer_pytorch_mlp.log`

Each is 40 to 60 lines and shows a short ReAct chain, typically looking at the dataset, training one model, and returning a one-row table as the Final Answer.

## Report snapshot at the time of this run

`report/benchmark_summary.md`:

```
# Benchmark Summary: 3 Algorithms x 2 Datasets (5-fold CV)
| Dataset | Algorithm | CV Mean Accuracy | CV Std | Test Accuracy | Source |
| iris | decision_tree | 0.9533 | 0.0340 | 0.9333 | agent |
| iris | random_forest | 0.9667 | 0.0211 | 0.9000 | agent |
| iris | pytorch_mlp | 0.8200 | 0.0452 | 0.7333 | agent |
| breast_cancer | decision_tree | 0.9209 | 0.0202 | 0.9386 | agent |
| breast_cancer | random_forest | 0.9543 | 0.0244 | 0.9561 | agent |
| breast_cancer | pytorch_mlp | 0.9508 | 0.0196 | 0.9386 | agent |
6/6 rows from live agent, 0/6 from deterministic fallback, agent_success_rate: 1.00 (6/6)
```

Captured 2026-09-03 15:29.
