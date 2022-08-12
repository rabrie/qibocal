# -*- coding: utf-8 -*-
"""Routine-specific functions for updating the platform runcard."""
import yaml


def resonator_spectroscopy_update(folder, qubit, freq, avg_voltage, peak_voltage):
    """Function to update platform runcard when executing a resonator spectroscopy"""

    with open(f"{folder}/platform.yml", "r") as file:
        settings = yaml.safe_load(file)

    settings["characterization"]["single_qubit"][qubit]["resonator_freq"] = int(freq)
    settings["characterization"]["single_qubit"][qubit][
        "resonator_spectroscopy_avg_ro_voltage"
    ] = float(avg_voltage)
    settings["characterization"]["single_qubit"][qubit]["state0_voltage"] = int(
        peak_voltage
    )

    with open(f"{folder}/data/resonator_spectroscopy/platform.yml", "w+") as file:
        yaml.dump(settings, file, sort_keys=False, indent=4)
