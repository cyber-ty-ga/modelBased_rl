"""Pack-level parameter identification utilities."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import optimize


class PackParameterIdentifier:
    """Identify pack-level parameters from experimental data."""

    @staticmethod
    def identify_from_pack_data(
        pack_data: pd.DataFrame,
        individual_cell_data: Optional[Dict[int, pd.DataFrame]] = None,
        n_cells: int = 20,
    ) -> Dict:
        results: Dict[str, Dict] = {}
        results["pack_ecm"] = PackParameterIdentifier._identify_pack_ecm(pack_data)
        results["pack_thermal"] = PackParameterIdentifier._identify_pack_thermal(pack_data)

        if individual_cell_data:
            results["cell_variations"] = PackParameterIdentifier._identify_cell_variations(
                individual_cell_data, n_cells
            )
        else:
            results["cell_variations"] = PackParameterIdentifier._estimate_variations_from_pack(
                pack_data, n_cells
            )

        if "cycle" in pack_data.columns:
            results["aging"] = PackParameterIdentifier._identify_aging_characteristics(pack_data)

        return results

    @staticmethod
    def _identify_pack_ecm(pack_data: pd.DataFrame) -> Dict:
        time = pack_data["time"].values
        voltage = pack_data["pack_voltage"].values
        current = pack_data["pack_current"].values

        current_diff = np.abs(np.diff(current))
        pulse_threshold = 0.5 * np.max(current_diff) if current_diff.size else 0.0
        pulse_indices = np.where(current_diff > pulse_threshold)[0]

        if len(pulse_indices) == 0:
            return {
                "R0_pack": 0.05,
                "R1_pack": 0.02,
                "C1_pack": 1000,
                "R2_pack": 0.03,
                "C2_pack": 5000,
            }

        pulse_idx = pulse_indices[0]
        segment = slice(max(0, pulse_idx - 10), min(len(time), pulse_idx + 100))

        V_segment = voltage[segment]
        I_segment = current[segment]

        V_before = np.mean(V_segment[:10])
        V_after = np.mean(V_segment[10:20])
        I_before = np.mean(I_segment[:10])
        I_after = np.mean(I_segment[10:20])

        delta_V = V_after - V_before
        delta_I = I_after - I_before

        R0_pack = abs(delta_V / delta_I) if abs(delta_I) > 0.1 else 0.05

        return {
            "R0_pack": float(R0_pack),
            "R1_pack": float(R0_pack * 0.4),
            "C1_pack": 1000.0,
            "R2_pack": float(R0_pack * 0.6),
            "C2_pack": 5000.0,
            "estimated_from_pulses": len(pulse_indices),
        }

    @staticmethod
    def _identify_pack_thermal(pack_data: pd.DataFrame) -> Dict:
        if "pack_temperature" not in pack_data.columns:
            return {"C_th_pack": 1500.0, "hA_pack": 10.0, "thermal_time_constant": 150.0}

        time = pack_data["time"].values
        temperature = pack_data["pack_temperature"].values
        current = pack_data["pack_current"].values

        heat_input = current**2
        heat_threshold = np.mean(heat_input) if heat_input.size else 0.0
        heating_segments = heat_input > heat_threshold

        if np.sum(heating_segments) < 10:
            return {"C_th_pack": 1500.0, "hA_pack": 10.0, "thermal_time_constant": 150.0}

        T_amb = np.min(temperature)

        try:
            segment_length = min(100, len(time))
            t_segment = time[:segment_length]
            T_segment = temperature[:segment_length]
            I_segment = current[:segment_length]
            R_heat = 0.02 * 20

            def thermal_model(t, C_th, hA):
                T_pred = np.zeros_like(t)
                T_pred[0] = T_segment[0]
                for i in range(1, len(t)):
                    dt = t[i] - t[i - 1]
                    Q_gen = I_segment[i] ** 2 * R_heat
                    Q_diss = hA * (T_pred[i - 1] - T_amb)
                    T_dot = (Q_gen - Q_diss) / C_th
                    T_pred[i] = T_pred[i - 1] + T_dot * dt
                return T_pred

            popt, _ = optimize.curve_fit(
                thermal_model,
                t_segment,
                T_segment,
                p0=[1500.0, 10.0],
                bounds=([100, 1], [10000, 100]),
                maxfev=5000,
            )

            C_th, hA = popt
            tau = C_th / hA
            return {"C_th_pack": float(C_th), "hA_pack": float(hA), "thermal_time_constant": float(tau)}

        except Exception:
            return {"C_th_pack": 1500.0, "hA_pack": 10.0, "thermal_time_constant": 150.0}

    @staticmethod
    def _identify_cell_variations(
        individual_cell_data: Dict[int, pd.DataFrame], n_cells: int
    ) -> Dict:
        variations = {
            "cell_ids": [],
            "Q_nominal": [],
            "R0": [],
            "initial_soc": [],
            "temperature_coefficients": [],
        }

        for cell_id, cell_data in individual_cell_data.items():
            if cell_id >= n_cells:
                continue
            variations["cell_ids"].append(cell_id)
            variations["Q_nominal"].append(2.5)
            variations["R0"].append(0.02)
            variations["initial_soc"].append(0.5)
            variations["temperature_coefficients"].append(0.001)

        stats = {
            "mean_Q": np.mean(variations["Q_nominal"]),
            "std_Q": np.std(variations["Q_nominal"]),
            "mean_R0": np.mean(variations["R0"]),
            "std_R0": np.std(variations["R0"]),
            "cv_Q": np.std(variations["Q_nominal"]) / np.mean(variations["Q_nominal"])
            if np.mean(variations["Q_nominal"]) > 0
            else 0,
            "cv_R0": np.std(variations["R0"]) / np.mean(variations["R0"])
            if np.mean(variations["R0"]) > 0
            else 0,
        }

        return {"individual": variations, "statistics": stats}

    @staticmethod
    def _estimate_variations_from_pack(pack_data: pd.DataFrame, n_cells: int) -> Dict:
        cell_voltage_columns = [
            col for col in pack_data.columns if "cell_" in col and "_voltage" in col
        ]

        if cell_voltage_columns:
            cell_voltages = pack_data[cell_voltage_columns].values
            voltage_std = np.std(cell_voltages, axis=1)
            mean_voltage_std = np.mean(voltage_std)
            estimated_capacity_cv = min(0.1, mean_voltage_std / 0.1)
            estimated_resistance_cv = estimated_capacity_cv * 2

            return {
                "estimated_from_voltage_spread": True,
                "voltage_std_mean": float(mean_voltage_std),
                "capacity_cv": float(estimated_capacity_cv),
                "resistance_cv": float(estimated_resistance_cv),
            }

        return {
            "estimated_from_voltage_spread": False,
            "capacity_cv": 0.05,
            "resistance_cv": 0.20,
            "source": "typical_manufacturing_tolerances",
        }

    @staticmethod
    def _identify_aging_characteristics(pack_data: pd.DataFrame) -> Dict:
        if "cycle" not in pack_data.columns:
            return {"available": False}

        cycles = pack_data["cycle"].unique()
        if len(cycles) < 10:
            return {"available": False, "reason": "insufficient_cycles"}

        aging_metrics = []
        for cycle in cycles:
            cycle_data = pack_data[pack_data["cycle"] == cycle]
            if len(cycle_data) > 10:
                max_voltage = cycle_data["pack_voltage"].max()
                min_voltage = cycle_data["pack_voltage"].min()
                voltage_range = max_voltage - min_voltage
                aging_metrics.append({"cycle": cycle, "voltage_range": voltage_range})

        if len(aging_metrics) > 5:
            cycles_array = [m["cycle"] for m in aging_metrics]
            voltage_ranges = [m["voltage_range"] for m in aging_metrics]
            slope, intercept = np.polyfit(cycles_array, voltage_ranges, 1)
            capacity_fade_rate = -slope / intercept if intercept != 0 else 0
            return {
                "available": True,
                "n_cycles": len(cycles),
                "capacity_fade_rate_per_cycle": float(capacity_fade_rate),
                "estimated_cycle_life": float(1.0 / capacity_fade_rate)
                if capacity_fade_rate > 0
                else 1000,
            }

        return {"available": False, "reason": "insufficient_data_for_trend"}
