"""Gym-like environment wrapper for pack-level H-AMBRL control."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from battery_pack_model import BatteryPack, PackConfiguration


class HAMBRLPackEnvironment:
    """Gym-like environment for H-AMBRL with battery pack."""

    def __init__(
        self,
        pack_config: PackConfiguration,
        max_steps: int = 1000,
        target_soc: float = 0.8,
        ambient_temp: float = 25.0,
        max_charge_current_a: float = 10.0,
        dt: float = 1.0,
    ):
        self.pack = BatteryPack(pack_config, dt=dt)
        self.max_steps = max_steps
        self.target_soc = target_soc
        self.ambient_temp = ambient_temp
        self.max_charge_current_a = max_charge_current_a
        self.dt = dt
        self.current_step = 0

        self.action_space = {"low": -1.0, "high": 1.0, "shape": (1,)}
        self.observation_space = {
            "low": np.array([0, 0, 0, 0, -100, 0]),
            "high": np.array([100, 100, 100, 100, 100, 1000]),
            "shape": (6,),
        }

    def reset(self, initial_soc: float = 0.2, temperature: float = 25.0) -> np.ndarray:
        self.pack.reset(initial_soc=initial_soc, temperature=temperature)
        self.current_step = 0
        return self._get_observation()

    def action_to_pack_current(self, action: float) -> float:
        action = float(np.clip(np.asarray(action).item(), -1.0, 1.0))
        # action=-1 -> 0 A, action=+1 -> -max_charge_current_a A (charging)
        return -0.5 * self.max_charge_current_a * (action + 1.0)

    def pack_current_to_action(self, pack_current: float) -> float:
        pack_current = float(np.clip(pack_current, -self.max_charge_current_a, 0.0))
        action = (-2.0 * pack_current / self.max_charge_current_a) - 1.0
        return float(np.clip(action, -1.0, 1.0))

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        # Map action [-1, 1] to charging current [0, -max_charge_current_a] A.
        # Current convention: positive = discharge, negative = charge.
        physical_current = self.action_to_pack_current(action)
        pack_state = self.pack.step(physical_current, ambient_temp=self.ambient_temp)
        reward = self._calculate_reward(pack_state)
        done = self._check_done(pack_state)
        self.current_step += 1
        observation = self._get_observation()
        return observation, reward, done, pack_state

    def _get_observation(self) -> np.ndarray:
        return np.array(
            [
                self.pack.pack_soc,
                self.pack.pack_voltage / 100,
                self.pack.pack_temperature / 100,
                self.pack.voltage_imbalance * 1000,
                self.pack.pack_current / 10,
                self.current_step / self.max_steps,
            ]
        )

    def _calculate_reward(self, pack_state: Dict) -> float:
        reward = 0.0
        reward += 10.0 * pack_state["pack_soc"]
        reward -= 0.01
        temp_penalty = max(0.0, pack_state["pack_temperature"] - 40) ** 2
        reward -= 0.1 * temp_penalty
        imbalance_penalty = pack_state["voltage_imbalance"] * 1000
        reward -= 0.05 * imbalance_penalty

        safety_events = pack_state.get("safety_events", {})
        n_events = sum(
            len(v) if isinstance(v, list) else (1 if v else 0) for v in safety_events.values()
        )
        reward -= 1.0 * n_events

        if pack_state["pack_soc"] >= self.target_soc:
            reward += 50.0

        return reward

    def _check_done(self, pack_state: Dict) -> bool:
        if self.current_step >= self.max_steps:
            return True
        if pack_state["pack_soc"] >= self.target_soc:
            return True

        safety_events = pack_state.get("safety_events", {})
        critical_violations = [
            safety_events.get("pack_over_voltage", False),
            safety_events.get("pack_under_voltage", False),
            len(safety_events.get("over_temperature_cells", [])) > 5,
        ]
        return any(critical_violations)
