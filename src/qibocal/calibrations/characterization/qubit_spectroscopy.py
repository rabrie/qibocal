import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Frequency", plots.frequency_msr_phase__fast_precision)
def qubit_spectroscopy(
    platform: AbstractPlatform,
    qubits: list,
    fast_start,
    fast_end,
    fast_step,
    precision_start,
    precision_end,
    precision_step,
    software_averages,
    points=10,
):

    platform.reload_settings()

    sequence = PulseSequence()

    qubit_frequencies = {}
    frequency_ranges = {}
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:

        qubit_frequencies[qubit] = platform.characterization["single_qubit"][qubit][
            "qubit_freq"
        ]
        frequency_ranges[qubit] = (
            np.arange(fast_start, fast_end, fast_step) + qubit_frequencies[qubit]
        )

        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=5000
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=5000)
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

        # FIXME: Waiting for Qblox platform to take care of that
        platform.ro_port[qubit].lo_frequency = (
            platform.characterization["single_qubit"][qubit]["resonator_freq"]
            - ro_pulses[qubit].frequency
        )

    data = DataUnits(
        name="fast_sweep", quantities={"frequency": "Hz"}, options=["qubit"]
    )

    # # FIXME: Waiting for Qblox platform to take care of that
    # platform.ro_port[qubit].lo_frequency = (
    #     platform.characterization["single_qubit"][qubit]["resonator_freq"]
    #     - ro_pulse.frequency
    # )
    count = 0
    for _ in range(software_averages):
        for freq in range(len(frequency_ranges[qubit])):  # FIXME: remove hardcoding
            if count % points == 0 and count > 0:
                yield data
                for qubit in qubits:
                    yield lorentzian_fit(
                        data.get_column("qubit", qubit),
                        x="frequency[GHz]",
                        y="MSR[uV]",
                        qubit=qubit,
                        nqubits=platform.settings["nqubits"],
                        labels=["qubit_freq", "peak_voltage"],
                    )

            for qubit in qubits:
                platform.qd_port[qubit].lo_frequency = (
                    frequency_ranges[qubit][freq] - qd_pulses[qubit].frequency
                )

            result = platform.execute_pulse_sequence(sequence)

            for qubit in qubits:
                msr, phase, i, q = result[ro_pulses[qubit].serial]

                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": frequency_ranges[qubit][freq],
                    "qubit": qubit,
                }
                data.add(results)
            count += 1
    yield data

    # if platform.resonator_type == "3D":
    #     qubit_frequency = data.get_values("frequency", "Hz")[
    #         np.argmin(data.get_values("MSR", "V"))
    #     ]
    #     avg_voltage = (
    #         np.mean(
    #             data.get_values("MSR", "V")[: ((fast_end - fast_start) // fast_step)]
    #         )
    #         * 1e6
    #     )
    # else:
    #     qubit_frequency = data.get_values("frequency", "Hz")[
    #         np.argmax(data.get_values("MSR", "V"))
    #     ]
    #     avg_voltage = (
    #         np.mean(
    #             data.get_values("MSR", "V")[: ((fast_end - fast_start) // fast_step)]
    #         )
    #         * 1e6
    #     )

    # prec_data = DataUnits(
    #     name=f"precision_sweep_q{qubit}", quantities={"frequency": "Hz"}
    # )
    # freqrange = (
    #     np.arange(precision_start, precision_end, precision_step) + qubit_frequency
    # )
    # count = 0
    # for _ in range(software_averages):
    #     for freq in freqrange:
    #         if count % points == 0 and count > 0:
    #             yield prec_data
    #             yield lorentzian_fit(
    #                 data + prec_data,
    #                 x="frequency[GHz]",
    #                 y="MSR[uV]",
    #                 qubit=qubit,
    #                 nqubits=platform.settings["nqubits"],
    #                 labels=["qubit_freq", "peak_voltage"],
    #             )
    #         platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
    #         msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
    #             ro_pulse.serial
    #         ]
    #         results = {
    #             "MSR[V]": msr,
    #             "i[V]": i,
    #             "q[V]": q,
    #             "phase[rad]": phase,
    #             "frequency[Hz]": freq,
    #         }
    #         prec_data.add(results)
    #         count += 1
    # yield prec_data
    # TODO: Estimate avg_voltage correctly


@plot("MSR and Phase vs Frequency", plots.frequency_flux_msr_phase)
def qubit_spectroscopy_flux(
    platform: AbstractPlatform,
    qubit: int,
    freq_width,
    freq_step,
    current_max,
    current_min,
    current_step,
    software_averages,
    fluxline,
    points=10,
):
    platform.reload_settings()

    if fluxline == "qubit":
        fluxline = qubit

    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=5000)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    data = DataUnits(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "current": "A"}
    )

    qubit_frequency = platform.characterization["single_qubit"][qubit]["qubit_freq"]
    qubit_biasing_current = platform.characterization["single_qubit"][qubit][
        "sweetspot"
    ]
    frequency_range = np.arange(-freq_width, freq_width, freq_step) + qubit_frequency
    current_range = (
        np.arange(current_min, current_max, current_step) + qubit_biasing_current
    )

    count = 0
    for _ in range(software_averages):
        for curr in current_range:
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
                platform.qf_port[fluxline].current = curr
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": freq,
                    "current[A]": curr,
                }
                # TODO: implement normalization
                data.add(results)
                count += 1

    yield data
