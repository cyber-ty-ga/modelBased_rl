"""Visualization utilities for pack simulation results."""
from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PackVisualizer:
    """Visualization tools for battery pack data."""

    @staticmethod
    def plot_pack_summary(results: pd.DataFrame, figsize: Tuple[int, int] = (16, 12)):
        fig, axes = plt.subplots(4, 3, figsize=figsize)

        ax = axes[0, 0]
        ax.plot(results["time"] / 60, results["pack_voltage"], "b-", label="Pack Voltage")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Pack Voltage (V)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

        ax2 = ax.twinx()
        ax2.plot(results["time"] / 60, results["pack_current"], "r-", label="Pack Current")
        ax2.set_ylabel("Current (A)")
        ax2.legend(loc="upper right")
        ax.set_title("Pack Voltage and Current")

        ax = axes[0, 1]
        ax.plot(results["time"] / 60, results["pack_soc"] * 100, "g-")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Pack SOC (%)")
        ax.grid(True, alpha=0.3)
        ax.set_title("Pack State of Charge")

        ax = axes[0, 2]
        cell_voltage_cols = [col for col in results.columns if "cell_" in col and "_voltage" in col]
        if cell_voltage_cols:
            cell_voltages = results[cell_voltage_cols]
            ax.plot(results["time"] / 60, cell_voltages.max(axis=1), "r-", label="Max Cell")
            ax.plot(results["time"] / 60, cell_voltages.min(axis=1), "b-", label="Min Cell")
            ax.plot(results["time"] / 60, cell_voltages.mean(axis=1), "g--", label="Average")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Cell Voltage (V)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title("Cell Voltage Distribution")

        ax = axes[1, 0]
        ax.plot(results["time"] / 60, results["voltage_imbalance"] * 1000, "m-")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Voltage Imbalance (mV)")
        ax.grid(True, alpha=0.3)
        ax.set_title("Cell-to-Cell Voltage Imbalance")

        ax = axes[1, 1]
        ax.plot(results["time"] / 60, results["soc_imbalance"] * 100, "c-")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("SOC Imbalance (%)")
        ax.grid(True, alpha=0.3)
        ax.set_title("Cell-to-Cell SOC Imbalance")

        ax = axes[1, 2]
        ax.plot(results["time"] / 60, results["pack_temperature"], "r-", label="Pack Max")
        ax.plot(results["time"] / 60, results["min_cell_temperature"], "b-", label="Cell Min")
        ax.plot(results["time"] / 60, results["ambient_temperature"], "g--", label="Ambient")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Temperature Profile")

        ax = axes[2, 0]
        ax.plot(results["time"] / 60, results["temperature_imbalance"], "orange")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Temperature Imbalance (°C)")
        ax.grid(True, alpha=0.3)
        ax.set_title("Cell-to-Cell Temperature Imbalance")

        ax = axes[2, 1]
        final_socs = [results[col].iloc[-1] for col in results.columns if "cell_" in col and "_soc" in col]
        if final_socs:
            ax.bar(range(len(final_socs)), final_socs)
            ax.set_xlabel("Cell Index")
            ax.set_ylabel("Final SOC")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Final SOC Distribution (Mean: {np.mean(final_socs):.3f})")

        ax = axes[2, 2]
        final_temps = [
            results[col].iloc[-1] for col in results.columns if "cell_" in col and "_temperature" in col
        ]
        if final_temps:
            ax.bar(range(len(final_temps)), final_temps)
            ax.set_xlabel("Cell Index")
            ax.set_ylabel("Final Temperature (°C)")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Final Temperature Distribution (Max: {max(final_temps):.1f}°C)")

        ax = axes[3, 0]
        safety_events = []
        times = []
        for _, row in results.iterrows():
            events = row.get("safety_events", {})
            if isinstance(events, dict):
                n_events = sum(
                    len(v) if isinstance(v, list) else (1 if v else 0)
                    for v in events.values()
                )
                if n_events > 0:
                    safety_events.append(n_events)
                    times.append(row["time"] / 60)
        if safety_events:
            ax.plot(times, safety_events, "ro-")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Safety Events Count")
            ax.grid(True, alpha=0.3)
            ax.set_title("Safety Events Timeline")

        ax = axes[3, 1]
        power = results["pack_voltage"] * results["pack_current"]
        ax.plot(results["time"] / 60, power / 1000, "purple")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Pack Power (kW)")
        ax.grid(True, alpha=0.3)
        ax.set_title("Pack Power")

        ax = axes[3, 2]
        energy = np.cumsum(power * np.gradient(results["time"])) / 3600 / 1000
        ax.plot(results["time"] / 60, energy, "brown")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Energy Throughput (kWh)")
        ax.grid(True, alpha=0.3)
        ax.set_title("Cumulative Energy")

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_cell_comparison(results: pd.DataFrame, cell_indices: List[int] | None = None):
        if cell_indices is None:
            cell_indices = [0, 5, 10, 15, 19]

        fig, axes = plt.subplots(3, 2, figsize=(14, 10))

        ax = axes[0, 0]
        for idx in cell_indices:
            col = f"cell_{idx}_voltage"
            if col in results.columns:
                ax.plot(results["time"] / 60, results[col], label=f"Cell {idx}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Voltage (V)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Selected Cell Voltages")

        ax = axes[0, 1]
        for idx in cell_indices:
            col = f"cell_{idx}_soc"
            if col in results.columns:
                ax.plot(results["time"] / 60, results[col] * 100, label=f"Cell {idx}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("SOC (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Selected Cell SOCs")

        ax = axes[1, 0]
        for idx in cell_indices:
            col = f"cell_{idx}_temperature"
            if col in results.columns:
                ax.plot(results["time"] / 60, results[col], label=f"Cell {idx}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Selected Cell Temperatures")

        ax = axes[1, 1]
        cell_voltage_cols = [col for col in results.columns if "cell_" in col and "_voltage" in col]
        if cell_voltage_cols:
            mean_voltage = results[cell_voltage_cols].mean(axis=1)
            for idx in cell_indices:
                col = f"cell_{idx}_voltage"
                if col in results.columns:
                    diff = results[col] - mean_voltage
                    ax.plot(results["time"] / 60, diff * 1000, label=f"Cell {idx}")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Voltage Difference from Mean (mV)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title("Cell Voltage Deviations")

        ax = axes[2, 0]
        final_socs = [results[col].iloc[-1] for col in results.columns if "cell_" in col and "_soc" in col]
        if final_socs:
            ax.hist(final_socs, bins=20, alpha=0.7, edgecolor="black")
            ax.set_xlabel("Final SOC")
            ax.set_ylabel("Number of Cells")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Final SOC Distribution (n={len(final_socs)} cells)")

        ax = axes[2, 1]
        for idx in cell_indices:
            soc_col = f"cell_{idx}_soc"
            volt_col = f"cell_{idx}_voltage"
            if soc_col in results.columns and volt_col in results.columns:
                ax.scatter(results[soc_col] * 100, results[volt_col], s=1, alpha=0.5, label=f"Cell {idx}")
        ax.set_xlabel("SOC (%)")
        ax.set_ylabel("Voltage (V)")
        ax.grid(True, alpha=0.3)
        ax.legend(markerscale=5)
        ax.set_title("Cell Voltage vs SOC")

        plt.tight_layout()
        return fig
