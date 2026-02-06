# modelBased_rl

## Baseline Benchmarking (CCCV vs MPC)

Run the two classical charging baselines on the same pack environment:

```bash
python scripts/run_baseline_benchmarks.py --objective safe --target-soc 0.8 --max-steps 1200
```

Supported objective presets come from `pack_experiments.py`:
- `fastest`
- `safe`
- `long_life`

Run all presets in one pass:

```bash
python scripts/run_baseline_benchmarks.py --objective all --target-soc 0.8 --max-steps 1200
```

Outputs are automatically separated into folders:
- `results/baselines/cccv`
- `results/baselines/mpc`
- `results/baselines/comparison`

Each folder includes:
- `trajectory.csv` (step-by-step simulation data)
- `metrics.json` or `metrics_summary.csv` (benchmark KPIs)
- high-resolution paper-style figures in both `.png` and `.pdf`

### Data-Calibrated Baselines (uses your real NASA/CALCE/MATR recordings)

Run CCCV vs MPC with environment settings calibrated from standardized CSVs and fitted parameter JSONs:

```bash
python scripts/run_baseline_benchmarks.py --use-real-data --dataset-families nasa,calce,matr --max-files-per-dataset 1 --objective safe --max-steps 1200
```

By default this mode treats those files as **cell-level behavior references** and scales them to your configured pack topology (`--n-series`, `--n-parallel`) so output figures are pack-focused.

When `--use-real-data` is enabled, outputs are written under:

- `results/baselines/data_calibrated/<objective>/<dataset_family>/<case>/cccv`
- `results/baselines/data_calibrated/<objective>/<dataset_family>/<case>/mpc`
- `results/baselines/data_calibrated/<objective>/<dataset_family>/<case>/comparison`

## Phase 1: Residual H-AMBRL Pipeline

1) Generate MPC-guided residual training data:

```bash
python scripts/generate_mpc_dataset.py --objective all --condition all --episodes-per-setting 2 --max-steps 1200
```

2) Train residual policy:

```bash
python scripts/train_residual_policy.py --dataset-csv data/training/residual_mpc_dataset.csv --model-out models/residual_hambrl_policy.json
```

3) Evaluate all controllers (CCCV vs MPC vs Residual):

```bash
python scripts/eval_all_controllers.py --model-path models/residual_hambrl_policy.json --objective all --condition all --max-steps 1200
```

Evaluation outputs are grouped by objective/condition under:
- `results/residual_phase1/evaluation/<objective>/<condition>/cccv`
- `results/residual_phase1/evaluation/<objective>/<condition>/mpc`
- `results/residual_phase1/evaluation/<objective>/<condition>/residual`
- `results/residual_phase1/evaluation/<objective>/<condition>/comparison`
