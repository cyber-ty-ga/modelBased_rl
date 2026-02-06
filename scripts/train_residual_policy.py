"""Train a phase-1 residual policy from the MPC dataset."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from controllers.residual_hambrl import ResidualHAMBRLController, ResidualPolicyConfig


@dataclass
class TrainConfig:
    dataset_csv: str = "data/training/residual_mpc_dataset.csv"
    model_out: str = "models/residual_hambrl_policy.json"
    report_json: str = "results/residual_phase1/training/training_report.json"
    diagnostics_dir: str = "results/residual_phase1/training/figures"
    val_ratio: float = 0.2
    random_seed: int = 123
    ridge_lambda: float = 1e-3
    delta_action_scale: float = 0.30
    max_delta_action: float = 0.40
    temp_soft_limit_c: float = 42.0


def parse_args() -> TrainConfig:
    import argparse

    parser = argparse.ArgumentParser(description="Train the residual H-AMBRL phase-1 policy.")
    parser.add_argument("--dataset-csv", type=str, default="data/training/residual_mpc_dataset.csv")
    parser.add_argument("--model-out", type=str, default="models/residual_hambrl_policy.json")
    parser.add_argument(
        "--report-json", type=str, default="results/residual_phase1/training/training_report.json"
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=str,
        default="results/residual_phase1/training/figures",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--delta-action-scale", type=float, default=0.30)
    parser.add_argument("--max-delta-action", type=float, default=0.40)
    parser.add_argument("--temp-soft-limit-c", type=float, default=42.0)
    args = parser.parse_args()
    return TrainConfig(
        dataset_csv=args.dataset_csv,
        model_out=args.model_out,
        report_json=args.report_json,
        diagnostics_dir=args.diagnostics_dir,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
        ridge_lambda=args.ridge_lambda,
        delta_action_scale=args.delta_action_scale,
        max_delta_action=args.max_delta_action,
        temp_soft_limit_c=args.temp_soft_limit_c,
    )


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 320,
        }
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (float(np.sum((y_true - y_pred) ** 2)) / denom if denom > 1e-12 else np.nan)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def pick_feature_columns(df: pd.DataFrame) -> List[str]:
    feature_cols = sorted(
        [c for c in df.columns if c.startswith("feature_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if not feature_cols:
        raise ValueError("No feature_* columns found in dataset.")
    return feature_cols


def save_figure(fig: plt.Figure, path_root: Path) -> None:
    path_root.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_root.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_root.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = parse_args()
    apply_plot_style()

    dataset_path = Path(config.dataset_csv)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    df = pd.read_csv(dataset_path)
    if df.empty:
        raise SystemExit("Dataset is empty.")
    if "target_delta_action" not in df.columns:
        raise SystemExit("Dataset missing required column: target_delta_action")

    feature_cols = pick_feature_columns(df)
    x = df[feature_cols].to_numpy(dtype=float)
    y = df["target_delta_action"].to_numpy(dtype=float)

    rng = np.random.default_rng(config.random_seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    split_idx = int((1.0 - config.val_ratio) * len(df))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    policy = ResidualHAMBRLController(
        config=ResidualPolicyConfig(
            delta_action_scale=config.delta_action_scale,
            max_delta_action=config.max_delta_action,
            temp_soft_limit_c=config.temp_soft_limit_c,
            ridge_lambda=config.ridge_lambda,
        )
    )

    fit_stats = policy.fit_supervised(x_train, y_train, ridge_lambda=config.ridge_lambda)
    train_pred = policy.predict_batch(x_train)
    val_pred = policy.predict_batch(x_val)
    train_metrics = regression_metrics(y_train, train_pred)
    val_metrics = regression_metrics(y_val, val_pred)

    model_out = Path(config.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    policy.save(model_out)

    diagnostics_dir = Path(config.diagnostics_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(y_train, bins=60, alpha=0.6, label="Train target", color="#457B9D")
    ax.hist(y_val, bins=60, alpha=0.6, label="Val target", color="#E76F51")
    ax.set_xlabel("Target residual delta action")
    ax.set_ylabel("Samples")
    ax.set_title("Residual Target Distribution")
    ax.legend()
    save_figure(fig, diagnostics_dir / "01_target_distribution")

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(y_val, val_pred, s=7, alpha=0.35, color="#2A9D8F", edgecolors="none")
    lim = float(max(np.max(np.abs(y_val)), np.max(np.abs(val_pred)), 0.05))
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="black", linewidth=1.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Target delta action")
    ax.set_ylabel("Predicted delta action")
    ax.set_title("Validation: Predicted vs Target")
    save_figure(fig, diagnostics_dir / "02_predicted_vs_target")

    residuals = y_val - val_pred
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(residuals, bins=70, color="#264653", alpha=0.8)
    ax.set_xlabel("Residual error (target - prediction)")
    ax.set_ylabel("Samples")
    ax.set_title("Validation Residual Error Distribution")
    save_figure(fig, diagnostics_dir / "03_residual_error")

    feature_table = pd.DataFrame(
        {
            "feature": feature_cols,
            "weight": policy.weights,
            "abs_weight": np.abs(policy.weights),
        }
    ).sort_values("abs_weight", ascending=False)
    feature_table.to_csv(diagnostics_dir / "feature_weights.csv", index=False)

    report = {
        "config": asdict(config),
        "dataset_path": str(dataset_path),
        "n_rows": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "feature_columns": feature_cols,
        "fit_stats": fit_stats,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "model_path": str(model_out),
        "feature_weights_csv": str((diagnostics_dir / "feature_weights.csv")),
    }
    report_path = Path(config.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)

    print("Trained residual policy.")
    print(f"Dataset rows: {len(df)}")
    print(f"Train RMSE: {train_metrics['rmse']:.6f}, Val RMSE: {val_metrics['rmse']:.6f}")
    print(f"Model saved to: {model_out.resolve()}")
    print(f"Training report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
