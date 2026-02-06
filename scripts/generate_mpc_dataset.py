"""Generate a supervised residual-learning dataset from MPC rollouts."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from battery_pack_model import PackConfiguration
from controllers.residual_hambrl import ResidualHAMBRLController
from hambrl_pack_env import HAMBRLPackEnvironment
from pack_experiments import build_default_conditions, build_default_objectives
from scripts.run_baseline_benchmarks import (
    MPCConfig,
    RolloutMPCController,
    count_safety_events,
    trim_pack_histories,
)


@dataclass
class DatasetConfig:
    output_csv: str = "data/training/residual_mpc_dataset.csv"
    output_meta_json: str = "data/training/residual_mpc_dataset_meta.json"
    objective: str = "all"
    condition: str = "all"
    episodes_per_setting: int = 2
    max_steps: int = 1200
    initial_soc: float = 0.2
    target_soc: float = 0.8
    n_series: int = 20
    n_parallel: int = 1
    max_charge_current_a: float = 10.0
    balancing_type: str = "passive"
    local_search_radius: float = 0.30
    local_search_points: int = 7
    random_seed: int = 123


def parse_args() -> DatasetConfig:
    import argparse

    objectives = build_default_objectives()
    conditions = build_default_conditions()
    parser = argparse.ArgumentParser(
        description="Generate phase-1 residual training data from MPC trajectories."
    )
    parser.add_argument("--output-csv", type=str, default="data/training/residual_mpc_dataset.csv")
    parser.add_argument(
        "--output-meta-json", type=str, default="data/training/residual_mpc_dataset_meta.json"
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
    parser.add_argument("--episodes-per-setting", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--initial-soc", type=float, default=0.2)
    parser.add_argument("--target-soc", type=float, default=0.8)
    parser.add_argument("--n-series", type=int, default=20)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--max-charge-current-a", type=float, default=10.0)
    parser.add_argument("--balancing-type", type=str, default="passive")
    parser.add_argument("--local-search-radius", type=float, default=0.30)
    parser.add_argument("--local-search-points", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=123)
    args = parser.parse_args()
    return DatasetConfig(
        output_csv=args.output_csv,
        output_meta_json=args.output_meta_json,
        objective=args.objective,
        condition=args.condition,
        episodes_per_setting=args.episodes_per_setting,
        max_steps=args.max_steps,
        initial_soc=args.initial_soc,
        target_soc=args.target_soc,
        n_series=args.n_series,
        n_parallel=args.n_parallel,
        max_charge_current_a=args.max_charge_current_a,
        balancing_type=args.balancing_type,
        local_search_radius=args.local_search_radius,
        local_search_points=args.local_search_points,
        random_seed=args.random_seed,
    )


def apply_aged_pack_state(env: HAMBRLPackEnvironment, initial_cycles: int) -> None:
    """Inject an aged state approximation by reducing capacity and increasing resistance."""
    if initial_cycles <= 0:
        return
    # 500 cycles -> roughly 12% fade, bounded for stability.
    capacity_fade = float(np.clip(initial_cycles / 500.0 * 0.12, 0.0, 0.30))
    for cell in env.pack.cells:
        cell.cycles = int(initial_cycles)
        cell.Q_loss = capacity_fade * 0.2 * cell.params.Q_nominal
        cell.Q_effective = cell.params.Q_nominal * (1.0 - capacity_fade)
        cell.R0_growth = cell.params.R0 * 2.0 * capacity_fade
    env.pack._update_pack_state()


def get_pack_state_snapshot(env: HAMBRLPackEnvironment) -> Dict:
    return {
        "pack_soc": env.pack.pack_soc,
        "pack_voltage": env.pack.pack_voltage,
        "pack_temperature": env.pack.pack_temperature,
        "voltage_imbalance": env.pack.voltage_imbalance,
        "pack_current": env.pack.pack_current,
        "safety_events": env.pack.safety_events,
    }


def local_action_score(
    env: HAMBRLPackEnvironment,
    action: float,
    cv_voltage_v: float,
    target_soc: float,
) -> float:
    """One-step heuristic score (lower is better) to build residual targets."""
    test_pack = copy.deepcopy(env.pack)
    pack_current = env.action_to_pack_current(action)
    next_state = test_pack.step(pack_current, ambient_temp=env.ambient_temp)

    soc_gap = max(0.0, target_soc - float(next_state["pack_soc"]))
    over_v = max(0.0, float(next_state["pack_voltage"]) - cv_voltage_v)
    over_t = max(0.0, float(next_state["pack_temperature"]) - 42.0)
    imbalance_mv = float(next_state["voltage_imbalance"]) * 1000.0
    current_a = -float(next_state["pack_current"])
    safety_count = count_safety_events(next_state.get("safety_events", {}))
    score = (
        120.0 * (soc_gap**2)
        + 60.0 * (over_v**2)
        + 30.0 * (over_t**2)
        + 0.003 * imbalance_mv
        + 0.01 * current_a
        + 300.0 * safety_count
    )
    return float(score)


def choose_local_target_action(
    env: HAMBRLPackEnvironment,
    mpc_action: float,
    cv_voltage_v: float,
    target_soc: float,
    radius: float,
    points: int,
) -> float:
    offsets = np.linspace(-radius, radius, points)
    best_action = float(np.clip(mpc_action, -1.0, 1.0))
    best_score = float("inf")
    for offset in offsets:
        candidate = float(np.clip(mpc_action + offset, -1.0, 1.0))
        score = local_action_score(
            env=env,
            action=candidate,
            cv_voltage_v=cv_voltage_v,
            target_soc=target_soc,
        )
        if score < best_score:
            best_score = score
            best_action = candidate
    return best_action


def main() -> None:
    config = parse_args()
    rng = np.random.default_rng(config.random_seed)

    objectives = build_default_objectives()
    conditions = build_default_conditions()

    objective_keys = list(objectives.keys()) if config.objective == "all" else [config.objective]
    condition_keys = list(conditions.keys()) if config.condition == "all" else [config.condition]

    rows: List[Dict] = []
    setting_index = 0

    for objective_key in objective_keys:
        objective = objectives[objective_key]
        for condition_key in condition_keys:
            condition = conditions[condition_key]
            for episode_idx in range(config.episodes_per_setting):
                pack_config = PackConfiguration(
                    n_series=config.n_series,
                    n_parallel=config.n_parallel,
                    balancing_type=config.balancing_type,
                )
                capacity_ah = pack_config.get_total_capacity()
                max_current_a = min(config.max_charge_current_a, objective.i_max_c_rate * capacity_ah)
                cv_voltage_v = objective.v_max * config.n_series

                env = HAMBRLPackEnvironment(
                    pack_config=pack_config,
                    max_steps=config.max_steps,
                    target_soc=config.target_soc,
                    ambient_temp=condition.ambient_temp,
                    max_charge_current_a=max_current_a,
                )
                init_soc_noise = float(rng.uniform(-0.015, 0.015))
                init_soc = float(np.clip(config.initial_soc + init_soc_noise, 0.05, 0.95))
                env.reset(initial_soc=init_soc, temperature=condition.ambient_temp)
                apply_aged_pack_state(env, condition.initial_cycles)
                trim_pack_histories(env.pack)

                mpc = RolloutMPCController(
                    config=MPCConfig(),
                    cv_voltage_v=cv_voltage_v,
                    max_charge_current_a=max_current_a,
                    target_soc=config.target_soc,
                )
                mpc.reset()

                state = get_pack_state_snapshot(env)
                for step_idx in range(config.max_steps):
                    mpc_action, mpc_info = mpc.act(state, env)
                    target_action = choose_local_target_action(
                        env=env,
                        mpc_action=mpc_action,
                        cv_voltage_v=cv_voltage_v,
                        target_soc=config.target_soc,
                        radius=config.local_search_radius,
                        points=config.local_search_points,
                    )
                    target_delta = float(np.clip(target_action - mpc_action, -1.0, 1.0))

                    features = ResidualHAMBRLController.build_features(
                        state=state,
                        target_soc=config.target_soc,
                        cv_voltage_v=cv_voltage_v,
                        max_charge_current_a=max_current_a,
                        temp_soft_limit_c=MPCConfig().temp_soft_limit_c,
                    )

                    _, reward, done, next_state = env.step(mpc_action)
                    trim_pack_histories(env.pack)
                    safety_count = count_safety_events(next_state.get("safety_events", {}))
                    row = {
                        "setting_id": setting_index,
                        "objective": objective_key,
                        "condition": condition_key,
                        "episode_idx": episode_idx,
                        "step_idx": step_idx,
                        "time_s": float(next_state["time"]),
                        "reward": float(reward),
                        "pack_soc": float(state["pack_soc"]),
                        "pack_voltage": float(state["pack_voltage"]),
                        "pack_temperature": float(state["pack_temperature"]),
                        "voltage_imbalance_v": float(state["voltage_imbalance"]),
                        "pack_current_a": float(state["pack_current"]),
                        "next_pack_soc": float(next_state["pack_soc"]),
                        "next_pack_voltage": float(next_state["pack_voltage"]),
                        "next_pack_temperature": float(next_state["pack_temperature"]),
                        "next_voltage_imbalance_v": float(next_state["voltage_imbalance"]),
                        "mpc_action": float(mpc_action),
                        "mpc_best_cost": float(mpc_info.get("mpc_best_cost", np.nan)),
                        "target_action": float(target_action),
                        "target_delta_action": float(target_delta),
                        "mpc_pack_current_a": float(env.action_to_pack_current(mpc_action)),
                        "max_charge_current_a": float(max_current_a),
                        "cv_voltage_v": float(cv_voltage_v),
                        "ambient_temp_c": float(condition.ambient_temp),
                        "target_soc": float(config.target_soc),
                        "safety_event_count": int(safety_count),
                    }
                    for feat_idx, feat_value in enumerate(features):
                        row[f"feature_{feat_idx}"] = float(feat_value)
                    rows.append(row)

                    state = next_state
                    if done:
                        break
                setting_index += 1

    if not rows:
        raise SystemExit("No rows generated. Check objective/condition selection.")

    dataset = pd.DataFrame(rows)
    output_csv = Path(config.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_csv, index=False)

    meta = {
        "config": asdict(config),
        "n_rows": int(len(dataset)),
        "n_settings": int(dataset["setting_id"].nunique()),
        "objectives": objective_keys,
        "conditions": condition_keys,
        "feature_dim": ResidualHAMBRLController.FEATURE_DIM,
        "columns": list(dataset.columns),
    }
    output_meta = Path(config.output_meta_json)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    with output_meta.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=True)

    print("Generated residual dataset.")
    print(f"Rows: {len(dataset)}")
    print(f"Output CSV: {output_csv.resolve()}")
    print(f"Output metadata: {output_meta.resolve()}")


if __name__ == "__main__":
    main()
