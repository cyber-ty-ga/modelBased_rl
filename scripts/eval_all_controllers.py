"""Evaluate CCCV, MPC, and residual-over-MPC controllers side by side."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from battery_pack_model import PackConfiguration
from controllers.residual_hambrl import ResidualHAMBRLController
from hambrl_pack_env import HAMBRLPackEnvironment
from pack_experiments import build_default_conditions, build_default_objectives
from scripts.run_baseline_benchmarks import (
    CCCVConfig,
    CCCVController,
    MPCConfig,
    RolloutMPCController,
    apply_publication_style,
    compute_metrics,
    plot_baseline_timeseries,
    plot_cell_statistics,
    plot_comparison_overlay,
    plot_metrics_bars,
    plot_phase_portraits,
    plot_tradeoff,
    save_json,
    trim_pack_histories,
)


@dataclass
class EvalConfig:
    model_path: str = "models/residual_hambrl_policy.json"
    output_root: str = "results/residual_phase1/evaluation"
    objective: str = "all"
    condition: str = "all"
    max_steps: int = 1200
    initial_soc: float = 0.2
    target_soc: float = 0.8
    n_series: int = 20
    n_parallel: int = 1
    max_charge_current_a: float = 10.0
    balancing_type: str = "passive"


def parse_args() -> EvalConfig:
    import argparse

    objectives = build_default_objectives()
    conditions = build_default_conditions()

    parser = argparse.ArgumentParser(description="Evaluate CCCV, MPC, and residual controllers.")
    parser.add_argument("--model-path", type=str, default="models/residual_hambrl_policy.json")
    parser.add_argument(
        "--output-root", type=str, default="results/residual_phase1/evaluation"
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=tuple(list(objectives.keys()) + ["all"]),
        default="all",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=tuple(list(conditions.keys()) + ["all"]),
        default="all",
    )
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--initial-soc", type=float, default=0.2)
    parser.add_argument("--target-soc", type=float, default=0.8)
    parser.add_argument("--n-series", type=int, default=20)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--max-charge-current-a", type=float, default=10.0)
    parser.add_argument("--balancing-type", type=str, default="passive")
    args = parser.parse_args()
    return EvalConfig(
        model_path=args.model_path,
        output_root=args.output_root,
        objective=args.objective,
        condition=args.condition,
        max_steps=args.max_steps,
        initial_soc=args.initial_soc,
        target_soc=args.target_soc,
        n_series=args.n_series,
        n_parallel=args.n_parallel,
        max_charge_current_a=args.max_charge_current_a,
        balancing_type=args.balancing_type,
    )


def apply_aged_pack_state(env: HAMBRLPackEnvironment, initial_cycles: int) -> None:
    if initial_cycles <= 0:
        return
    capacity_fade = min(0.30, initial_cycles / 500.0 * 0.12)
    for cell in env.pack.cells:
        cell.cycles = int(initial_cycles)
        cell.Q_loss = capacity_fade * 0.2 * cell.params.Q_nominal
        cell.Q_effective = cell.params.Q_nominal * (1.0 - capacity_fade)
        cell.R0_growth = cell.params.R0 * 2.0 * capacity_fade
    env.pack._update_pack_state()


def initial_state(env: HAMBRLPackEnvironment) -> Dict:
    return {
        "pack_soc": env.pack.pack_soc,
        "pack_voltage": env.pack.pack_voltage,
        "pack_temperature": env.pack.pack_temperature,
        "voltage_imbalance": env.pack.voltage_imbalance,
        "pack_current": env.pack.pack_current,
        "safety_events": env.pack.safety_events,
    }


class ResidualOverMPCController:
    """Residual policy stacked over a base MPC controller."""

    def __init__(
        self,
        residual_policy: ResidualHAMBRLController,
        mpc_controller: RolloutMPCController,
        cv_voltage_v: float,
    ) -> None:
        self.residual_policy = residual_policy
        self.mpc_controller = mpc_controller
        self.cv_voltage_v = cv_voltage_v

    def reset(self) -> None:
        self.mpc_controller.reset()

    def act(self, state: Dict, env: HAMBRLPackEnvironment):
        mpc_action, mpc_info = self.mpc_controller.act(state, env)
        action, residual_info = self.residual_policy.act(
            state=state,
            env=env,
            mpc_action=mpc_action,
            cv_voltage_v=self.cv_voltage_v,
        )
        info = {
            "controller_mode": "RESIDUAL",
            "mpc_action": float(mpc_action),
            "desired_pack_current": float(env.action_to_pack_current(action)),
            "shield_used": bool(residual_info.get("shield_used", False)),
            "delta_action": float(residual_info.get("delta_action", 0.0)),
            "mpc_best_cost": float(mpc_info.get("mpc_best_cost", float("nan"))),
        }
        return action, info


def run_episode(
    env: HAMBRLPackEnvironment,
    controller,
    controller_name: str,
    initial_soc: float,
    condition,
) -> pd.DataFrame:
    env.reset(initial_soc=initial_soc, temperature=condition.ambient_temp)
    apply_aged_pack_state(env, condition.initial_cycles)
    trim_pack_histories(env.pack)
    controller.reset()

    rows: List[Dict] = []
    state = initial_state(env)
    for _ in range(env.max_steps):
        action, info = controller.act(state, env)
        _, reward, done, next_state = env.step(action)
        trim_pack_histories(env.pack)

        row = dict(next_state)
        row["controller"] = controller_name
        row["action"] = float(action)
        row["reward"] = float(reward)
        for key, value in info.items():
            row[key] = value
        rows.append(row)

        state = next_state
        if done:
            break
    return pd.DataFrame(rows)


def main() -> None:
    config = parse_args()
    apply_publication_style()

    model_path = Path(config.model_path)
    if not model_path.exists():
        raise SystemExit(f"Residual model not found: {model_path}")
    residual_policy = ResidualHAMBRLController.load(model_path)

    objectives = build_default_objectives()
    conditions = build_default_conditions()
    objective_keys = list(objectives.keys()) if config.objective == "all" else [config.objective]
    condition_keys = list(conditions.keys()) if config.condition == "all" else [config.condition]

    summary_rows: List[Dict] = []
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for objective_key in objective_keys:
        objective = objectives[objective_key]
        for condition_key in condition_keys:
            condition = conditions[condition_key]

            pack_config = PackConfiguration(
                n_series=config.n_series,
                n_parallel=config.n_parallel,
                balancing_type=config.balancing_type,
            )
            capacity_ah = pack_config.get_total_capacity()
            max_current_a = min(config.max_charge_current_a, objective.i_max_c_rate * capacity_ah)
            cv_voltage_v = objective.v_max * config.n_series

            cccv = CCCVController(
                config=CCCVConfig(),
                cv_voltage_v=cv_voltage_v,
                max_charge_current_a=max_current_a,
                target_soc=config.target_soc,
            )
            mpc = RolloutMPCController(
                config=MPCConfig(),
                cv_voltage_v=cv_voltage_v,
                max_charge_current_a=max_current_a,
                target_soc=config.target_soc,
            )
            residual_ctrl = ResidualOverMPCController(
                residual_policy=copy.deepcopy(residual_policy),
                mpc_controller=RolloutMPCController(
                    config=MPCConfig(),
                    cv_voltage_v=cv_voltage_v,
                    max_charge_current_a=max_current_a,
                    target_soc=config.target_soc,
                ),
                cv_voltage_v=cv_voltage_v,
            )
            controllers = {
                "cccv": cccv,
                "mpc": mpc,
                "residual": residual_ctrl,
            }

            setting_root = output_root / objective_key / condition_key
            comparison_root = setting_root / "comparison"
            comparison_root.mkdir(parents=True, exist_ok=True)

            all_results: Dict[str, pd.DataFrame] = {}
            all_metrics: Dict[str, Dict] = {}

            for name, controller in controllers.items():
                env = HAMBRLPackEnvironment(
                    pack_config=pack_config,
                    max_steps=config.max_steps,
                    target_soc=config.target_soc,
                    ambient_temp=condition.ambient_temp,
                    max_charge_current_a=max_current_a,
                )
                results = run_episode(
                    env=env,
                    controller=controller,
                    controller_name=name,
                    initial_soc=config.initial_soc,
                    condition=condition,
                )
                all_results[name] = results
                metrics = compute_metrics(results, target_soc=config.target_soc)
                all_metrics[name] = metrics

                algo_root = setting_root / name
                algo_root.mkdir(parents=True, exist_ok=True)
                results.to_csv(algo_root / "trajectory.csv", index=False)
                save_json(
                    algo_root / "metrics.json",
                    {
                        "objective": objective_key,
                        "condition": condition_key,
                        "algorithm": name,
                        "metrics": metrics,
                        "cv_voltage_v": cv_voltage_v,
                        "max_charge_current_a": max_current_a,
                    },
                )
                plot_baseline_timeseries(
                    results=results,
                    controller_name=name.upper(),
                    target_soc=config.target_soc,
                    cv_voltage_v=cv_voltage_v,
                    output_path=algo_root / "figures" / "01_timeseries",
                )
                plot_cell_statistics(
                    results=results,
                    controller_name=name.upper(),
                    output_path=algo_root / "figures" / "02_cell_statistics",
                )
                plot_phase_portraits(
                    results=results,
                    controller_name=name.upper(),
                    output_path=algo_root / "figures" / "03_phase_portraits",
                )

                summary_row = {
                    "objective": objective_key,
                    "condition": condition_key,
                    "controller": name,
                }
                summary_row.update(metrics)
                summary_rows.append(summary_row)

            metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
            metrics_df.index.name = "controller"
            metrics_df.to_csv(comparison_root / "metrics_summary.csv")
            save_json(
                comparison_root / "run_metadata.json",
                {
                    "eval_config": asdict(config),
                    "objective_key": objective_key,
                    "objective": asdict(objective),
                    "condition_key": condition_key,
                    "condition": asdict(condition),
                    "cv_voltage_v": cv_voltage_v,
                    "max_charge_current_a": max_current_a,
                },
            )
            plot_comparison_overlay(
                all_results=all_results,
                target_soc=config.target_soc,
                cv_voltage_v=cv_voltage_v,
                output_path=comparison_root / "figures" / "01_overlay",
            )
            plot_metrics_bars(
                metrics_df=metrics_df,
                output_path=comparison_root / "figures" / "02_metrics_bar",
            )
            plot_tradeoff(
                metrics_df=metrics_df,
                output_path=comparison_root / "figures" / "03_tradeoff",
            )
            print(
                f"Completed evaluation for objective={objective_key}, condition={condition_key}"
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_root / "summary_all_settings.csv"
    summary_df.to_csv(summary_path, index=False)
    print("Completed all evaluations.")
    print(f"Summary CSV: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
