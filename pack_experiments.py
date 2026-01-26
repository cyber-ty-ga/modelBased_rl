"""Scenario definitions and experiment matrix for pack charging studies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ChargingObjective:
    name: str
    description: str
    v_max: float
    t_max: float
    i_max_c_rate: float
    weight_time: float = 1.0
    weight_q_loss: float = 0.0


@dataclass(frozen=True)
class BatteryCondition:
    name: str
    description: str
    ambient_temp: float
    initial_cycles: int = 0


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str
    description: str


@dataclass(frozen=True)
class ScenarioDefinition:
    condition: BatteryCondition
    objective: ChargingObjective
    algorithm: AlgorithmConfig
    label: str


def build_default_objectives() -> Dict[str, ChargingObjective]:
    return {
        "fastest": ChargingObjective(
            name="Fastest Possible",
            description="Minimize time to 80% SOC with relaxed safety limits.",
            v_max=4.2,
            t_max=45.0,
            i_max_c_rate=4.0,
        ),
        "safe": ChargingObjective(
            name="Safe",
            description="Maintain conservative limits with moderate charge time.",
            v_max=4.15,
            t_max=40.0,
            i_max_c_rate=3.0,
        ),
        "long_life": ChargingObjective(
            name="Long-life",
            description="Prioritize degradation minimization with lower limits.",
            v_max=4.1,
            t_max=35.0,
            i_max_c_rate=2.0,
            weight_q_loss=10.0,
        ),
    }


def build_default_conditions() -> Dict[str, BatteryCondition]:
    return {
        "fresh": BatteryCondition(
            name="Fresh Cell",
            description="0-cycle cell at 25°C.",
            ambient_temp=25.0,
        ),
        "aged": BatteryCondition(
            name="Aged Cell",
            description="500-cycle cell at 25°C.",
            ambient_temp=25.0,
            initial_cycles=500,
        ),
        "cold": BatteryCondition(
            name="Cold Ambient",
            description="0°C start with fresh cells.",
            ambient_temp=0.0,
        ),
    }


def build_default_algorithms() -> Dict[str, AlgorithmConfig]:
    return {
        "cccv": AlgorithmConfig(
            name="CCCV Baseline",
            description="Constant-current/constant-voltage charging.",
        ),
        "mpc": AlgorithmConfig(
            name="Physics-based MPC",
            description="ECM + thermal model with MPC optimization.",
        ),
        "hambrl": AlgorithmConfig(
            name="H-AMBRL",
            description="Hybrid adaptive model-based reinforcement learning.",
        ),
    }


def build_experiment_matrix() -> List[ScenarioDefinition]:
    objectives = build_default_objectives()
    conditions = build_default_conditions()
    algorithms = build_default_algorithms()

    labels = {
        ("fresh", "fastest"): "F1",
        ("fresh", "safe"): "S1",
        ("fresh", "long_life"): "L1",
        ("aged", "fastest"): "F2",
        ("aged", "safe"): "S2",
        ("aged", "long_life"): "L2",
        ("cold", "fastest"): "F3",
        ("cold", "safe"): "S3",
        ("cold", "long_life"): "L3",
    }

    scenarios: List[ScenarioDefinition] = []
    for condition_key, condition in conditions.items():
        for objective_key, objective in objectives.items():
            for algorithm in algorithms.values():
                label = labels[(condition_key, objective_key)]
                scenarios.append(
                    ScenarioDefinition(
                        condition=condition,
                        objective=objective,
                        algorithm=algorithm,
                        label=label,
                    )
                )

    return scenarios
